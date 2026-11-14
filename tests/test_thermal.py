"""GPU temperature control: a simulated card that heats with work and cools while idle settles in the band."""
import os
import unittest
from unittest import mock

from apsi import thermal


class FakeCard:
    """Heats toward 92 C at full load, cools toward 35 C idle (first-order, time constant tau seconds)."""

    def __init__(self, clock):
        self.clock, self.temp, self.busy = clock, 60.0, 0.0
        self.last = clock[0]

    def advance(self, dt, working):
        tau = 6.0
        goal = 92.0 if working else 35.0
        self.temp += (goal - self.temp) * min(1.0, dt / tau)
        self.busy += dt if working else 0.0


class Thermal(unittest.TestCase):
    def run_sim(self, target, seconds=240.0):
        clock = [0.0]
        card = FakeCard(clock)
        with mock.patch.object(thermal.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(thermal.time, "sleep", lambda s: (card.advance(s, False), clock.__setitem__(0, clock[0] + s))):
            g = thermal.Governor(lambda: card.temp, target)
            temps = []
            while clock[0] < seconds:
                step = 0.05                                 # a training step / a token
                card.advance(step, True)
                clock[0] += step
                g.tick()
                temps.append((clock[0], card.temp))
        return g, card, temps

    def test_settles_in_the_band(self):
        g, card, temps = self.run_sim(77.0)
        late = [t for c, t in temps if c > 120]
        self.assertLess(max(late), 79.5)
        self.assertGreater(min(late), 73.0)
        self.assertGreater(card.busy / 240.0, 0.4)          # still does most of the work
        self.assertGreater(g.max_temp, 78.0)                # it did have to act

    def test_cool_card_is_never_slowed(self):
        g, card, temps = self.run_sim(95.0, 60.0)
        self.assertEqual(g.paced, 0.0)

    def test_off_and_unknown(self):
        with mock.patch.dict(os.environ, {"APSI_GPU_TEMP": "off"}):
            self.assertIs(thermal.governor(object()), thermal.NULL)
        self.assertIs(thermal.governor(None), thermal.NULL)
        thermal.NULL.tick()


class SimulatedCard(unittest.TestCase):
    """A card whose die jumps with every change of load (a fast part, 0.3 s) on top of the slow heating of the cooler
    (40 s), read in whole degrees: the pacing must hold it level, not swing between paused and hot."""

    def run_card(self, slow_gain, secs=300.0, step=0.028):
        import math
        import random
        from apsi import thermal as th
        rnd = random.Random(0)

        class Clock:
            t = 0.0

            def monotonic(self):
                return self.t

            def sleep(self, d):
                self.t += d
        clk = Clock()
        s = {"f": 0.0, "s": slow_gain * 0.75}

        def heat(dt, on):
            s["f"] += (15.0 * on - s["f"]) * (1 - math.exp(-dt / 0.3))
            s["s"] += (slow_gain * on - s["s"]) * (1 - math.exp(-dt / 40.0))
        with mock.patch.object(th, "time", clk):
            g = th.Governor(lambda: float(round(35 + s["f"] + s["s"] + rnd.uniform(-0.3, 0.3))), 77)
            samples, last, work = [], 0.0, 0.0
            while clk.t < secs:
                clk.t += step
                work += step
                heat(step, 1.0)
                b = clk.t
                g.tick(lambda: None)
                p = clk.t - b
                for _ in range(max(1, int(p / 0.02))):
                    heat(p / max(1, int(p / 0.02)), 0.0)
                if clk.t - last >= 2.0:
                    samples.append((35 + s["f"] + s["s"], work / (clk.t - last)))
                    last, work = clk.t, 0.0
        return samples[len(samples) // 3:]

    def test_level_temperature_and_speed(self):
        for gain in (30.0, 36.0):
            tail = self.run_card(gain)
            temps, speeds = [t for t, _ in tail], [v for _, v in tail]
            self.assertGreater(min(temps), 74.5)
            self.assertLess(max(temps), 78.5)
            self.assertLess(max(speeds) - min(speeds), 0.05)          # steady speed, no swings
        self.assertGreater(min(v for _, v in self.run_card(30.0)), 0.9)

    def test_a_cool_card_is_never_slowed(self):
        self.assertAlmostEqual(min(v for _, v in self.run_card(20.0)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class PowerLimitRestore(unittest.TestCase):
    """A run that was killed never put its lowered power limit back: the next run must restore the card's own limit
    (recorded by the first change), not take the low one for the card's own."""

    def test_killed_run_is_undone(self):
        import ctypes
        import tempfile

        class Lib:
            def __init__(self):
                self.limit, self.default = 136000, 170000

            def nvmlDeviceGetPowerManagementLimit(self, h, ref):
                ref._obj.value = self.limit
                return 0

            def nvmlDeviceGetPowerManagementDefaultLimit(self, h, ref):
                ref._obj.value = self.default
                return 0

            def nvmlDeviceGetPowerManagementLimitConstraints(self, h, lo, hi):
                lo._obj.value, hi._obj.value = 100000, 170000
                return 0

            def nvmlDeviceSetPowerManagementLimit(self, h, v):
                self.limit = v.value
                return 0

        with tempfile.TemporaryDirectory() as d, mock.patch.object(thermal, "_power_state_file",
                                                                   lambda i: os.path.join(d, "p.txt")), \
                mock.patch("atexit.register"):
            lib = Lib()
            pc = thermal._PowerControl.try_make(lib, None, 0)
            pc.set(104000)                               # the first run lowers the limit and is killed: no restore
            self.assertEqual(lib.limit, 104000)
            pc2 = thermal._PowerControl.try_make(lib, None, 0)
            self.assertEqual(lib.limit, 136000)          # the next run starts from the card's own limit
            self.assertEqual(pc2.orig, 136000)
            pc2.set(110000)
            pc2.restore()                                # a clean end: the limit back, the record gone
            self.assertEqual(lib.limit, 136000)
            self.assertFalse(os.path.exists(os.path.join(d, "p.txt")))


class PowerLimitControl(unittest.TestCase):
    """The power limit holds a card at 78-79 C under full load and never lets it reach 80, through a light phase
    and back (sudden changes of load), for slow and fast coolers and hot and cool rooms."""

    def run_card(self, tau=20.0, amb=32.0, seconds=400.0):
        clock, st = [0.0], {"T": 45.0, "limit": 136000.0}

        def demand(t):
            return 40.0 if t < 10 else 90.0 if 150 < t < 170 else 175.0

        def power():
            return min(st["limit"] / 1000.0, demand(clock[0]))

        def read():
            return float(round(st["T"] + 0.035 * power()))

        class PC:
            top, floor, limit = 170000, 93500, 136000.0

            def set(self, mw):
                st["limit"] = float(mw)
        pc = PC()
        pc.step = thermal._PowerControl.step.__get__(pc)
        g = thermal.Governor(read, 79.0)
        g.power, g.read_power = pc, power
        temps, watts = [], []
        with mock.patch.object(thermal.time, "monotonic", lambda: clock[0]):
            while clock[0] < seconds:
                st["T"] += (amb + 0.30 * power() - st["T"]) * min(1.0, 0.05 / tau)
                clock[0] += 0.05
                g.tick()
                temps.append((clock[0], read()))
                watts.append((clock[0], power()))
        return temps, watts, g

    def test_steady_and_never_80(self):
        for tau, amb in ((20.0, 32.0), (8.0, 32.0), (40.0, 32.0), (20.0, 38.0), (20.0, 26.0)):
            temps, watts, g = self.run_card(tau, amb)
            full = [t for c, t in temps if 90 < c < 150 or c > 230]
            self.assertLess(max(t for c, t in temps), 80.0, (tau, amb))
            self.assertGreaterEqual(min(full), 77.0, (tau, amb))    # (changes gathered: 1 W, 1 s apart)
            self.assertEqual(g.paced, 0.0)
            # as much power as 78.5 C allows (the steady state of this card: amb + 0.335 P = 78.5)
            steady = (78.5 - amb) / 0.335
            late = [p for c, p in watts if c > 230]
            self.assertGreater(sum(late) / len(late), steady * 0.97, (tau, amb))


class Retarget(unittest.TestCase):
    def test_a_reused_governor_keeps_the_ceiling(self):
        """A second run in the same program (a warm-up fit, then the training) reuses the governor: it must keep
        steering to ceiling - 0.5 in [ceiling - 1, ceiling], never a band that reaches 80."""
        g = thermal.Governor(lambda: 70.0, 79.0)
        g.retarget(79.0)
        self.assertEqual((g.target, g.low, g.high), (78.5, 78.0, 79.0))


class FanControl(unittest.TestCase):
    """A fixed 145 W; the fans (70-82 %) hold the temperature; the power limit only when the fans are at the top."""

    @staticmethod
    def _sim(seconds, ambient, r_lo, r_hi, load=None):
        """A card at a fixed 145 W limit: temperature rise per watt falls from r_lo (fans at 0 %) to r_hi (100 %);
        the fans follow their command in about 2 s, the die in about 20 s; the sensor reads whole degrees."""
        clock, st = [0.0], {"T": 50.0, "limit": 145000.0, "cmd": None, "fan": 30.0}
        log = []

        def power():
            frac = 1.0 if load is None else load(clock[0])
            return min(st["limit"] / 1000.0, 145.0 * frac)

        def read():
            return float(round(st["T"]))

        class PC:
            top, floor, orig, limit, run = 170000, 93500, 136000, 145000.0, 145000.0

            def set(self, mw):
                st["limit"] = float(mw)
        pc = PC()
        pc.step = thermal._PowerControl.step.__get__(pc)
        g = thermal.Governor(read, 79.0)
        g.power, g.read_power = pc, power
        fc = thermal._FanControl()
        fc.percent, fc.level, fc.ok = None, None, True
        fc.low, fc.top = 70.0, 82.0
        fc._set = lambda pct: st.__setitem__("cmd", pct) or True
        g.fan = fc
        with mock.patch.object(thermal.time, "monotonic", lambda: clock[0]):
            while clock[0] < seconds:
                want = 30.0 + 0.4 * max(0.0, st["T"] - 50.0) if st["cmd"] is None else float(st["cmd"])
                st["fan"] += (want - st["fan"]) * min(1.0, 0.05 / 2.0)
                r = r_lo + (r_hi - r_lo) * st["fan"] / 100.0
                st["T"] += (ambient + r * power() - st["T"]) * min(1.0, 0.05 / 20.0)
                clock[0] += 0.05
                g.tick()
                log.append((clock[0], read(), st["limit"], st["fan"], power(), st["cmd"]))
        fc.auto()
        return log, st

    def test_fixed_power_fans_hold_ceiling(self):
        log, st = self._sim(600.0, 35.0, 0.44, 0.24)
        self.assertTrue(all(x[1] < 80.0 for x in log))
        late = [x for x in log if x[0] > 200.0]
        self.assertTrue(all(70.0 <= x[1] <= 79.0 for x in late))
        self.assertTrue(all(x[2] == 145000.0 for x in log))                    # the power never moved
        self.assertTrue(all(70 <= x[5] <= 82 for x in late if x[5] is not None))
        self.assertIsNone(st["cmd"])                                          # driver's curve at the end

    def test_sudden_load_steps(self):
        """Light and heavy phases every 40 s: still never 80, power untouched."""
        log, _ = self._sim(600.0, 35.0, 0.44, 0.24, load=lambda t: 1.0 if int(t / 40) % 2 == 0 else 0.55)
        self.assertTrue(all(x[1] < 80.0 for x in log))
        self.assertTrue(all(x[2] == 145000.0 for x in log))

    def test_cool_card_driver_curve(self):
        log, st = self._sim(300.0, 22.0, 0.30, 0.20)
        self.assertIsNone(st["cmd"])
        self.assertTrue(all(x[2] == 145000.0 for x in log))

    def test_fans_at_top_then_power_as_last_resort(self):
        """A hot room: fans at 82 % cannot hold 79 at 145 W; the limit comes down only then, slowly, and the card
        stays under 80 once the controller has it."""
        log, _ = self._sim(900.0, 40.0, 0.44, 0.30)
        late = [x for x in log if x[0] > 300.0]
        self.assertTrue(all(x[1] < 80.0 for x in late))
        self.assertLess(min(x[2] for x in late), 145000.0)
        self.assertLess(max(x[2] for x in late) - min(x[2] for x in late), 12000.0)
        self.assertGreater(min(x[5] for x in late if x[5] is not None), 75.0)

    def test_train_power(self):
        class PC:
            top, limit = 170000, 136000.0
        with mock.patch.dict(os.environ, {"APSI_GPU_TRAIN_POWER": ""}, clear=False):
            os.environ.pop("APSI_GPU_TRAIN_POWER")
            self.assertEqual(thermal._train_power(PC()), 144500.0)            # 85 % of the default
            self.assertEqual(thermal._train_power(PC(), {"power_w": 150.0}), 150000.0)   # the measured setting
        with mock.patch.dict(os.environ, {"APSI_GPU_TRAIN_POWER": "140W"}):
            self.assertEqual(thermal._train_power(PC()), 140000.0)
        with mock.patch.dict(os.environ, {"APSI_GPU_TRAIN_POWER": "200W"}):
            self.assertEqual(thermal._train_power(PC()), 170000.0)        # never above the default
        with mock.patch.dict(os.environ, {"APSI_GPU_TRAIN_POWER": "100W"}):
            self.assertEqual(thermal._train_power(PC()), 136000.0)        # never below the card's own limit


class Calibrate(unittest.TestCase):
    """apsi gpu calibrate on a simulated card: the fitted settled temperatures and the plan."""

    def _card(self, ambient, r_lo, r_hi):
        from apsi import gpu_calibrate as gc
        st = {"t": 0.0, "T": ambient + 8.0, "P": 136.0, "fan": 40.0, "load": False, "cmd": None}

        def adv(dt):
            steps = int(dt / 0.05)
            for _ in range(steps):
                want = 40.0 if st["cmd"] is None else st["cmd"]
                st["fan"] += (want - st["fan"]) * 0.025
                r = r_lo + (r_hi - r_lo) * st["fan"] / 100.0
                p = st["P"] if st["load"] else 12.0
                st["T"] += (ambient + r * p - st["T"]) * 0.05 / 20.0
                st["t"] += 0.05
        args = dict(read_temp=lambda: float(round(st["T"])), read_power=lambda: st["P"] if st["load"] else 12.0,
                    set_power=lambda w: st.__setitem__("P", w), set_fan=lambda f: st.__setitem__("cmd", f),
                    start_load=lambda: st.__setitem__("load", True), stop_load=lambda: st.__setitem__("load", False),
                    top_w=170.0, own_w=136.0, sleep=adv, now=lambda: st["t"], say=lambda m: None)
        return gc, st, args

    def test_cool_card_gets_more_power(self):
        gc, st, args = self._card(28.0, 0.40, 0.20)
        prof = gc.measure(**args)
        r82 = 0.40 - 0.20 * 0.82
        best = (77.5 - (28.0 + 12.0 * (0.40 - 0.2 * 0.4))) / r82
        self.assertGreater(prof["power_w"], 136.0)
        self.assertLessEqual(prof["power_w"], 170.0)
        self.assertLess(abs(prof["power_w"] - min(170.0, 0.97 * best)), 25.0)
        self.assertIsNone(st["cmd"])                                      # the driver's fan curve is back
        self.assertFalse(st["load"])

    def test_hot_card_widens_the_fans(self):
        gc, st, args = self._card(38.0, 0.46, 0.30)
        prof = gc.measure(**args)
        self.assertGreaterEqual(prof["power_w"], 136.0)
        self.assertGreater(prof["fan_max"], 82)

    def test_fit(self):
        from apsi import gpu_calibrate as gc
        import math
        ts = [i * 0.5 for i in range(140)]
        temps = [70.0 + (50.0 - 70.0) * math.exp(-t / 18.0) for t in ts]
        end, tau = gc.fit(ts, temps)
        self.assertAlmostEqual(end, 70.0, delta=0.3)
        self.assertAlmostEqual(tau, 18.0, delta=2.0)

    def test_profile_saved_and_read(self):
        import tempfile
        from apsi import gpu_calibrate as gc
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"APSI_HOME": d}):
            gc.save_profile("0:RTX", {"power_w": 150.0, "fan_min": 70, "fan_max": 84})
            self.assertEqual(gc.load_profile("0:RTX")["fan_max"], 84)
            self.assertIsNone(gc.load_profile("1:other"))
            self.assertEqual(thermal._fan_range(gc.load_profile("0:RTX")), (70.0, 84.0))


class CalibrateNoFans(Calibrate):
    def test_driver_fans_only(self):
        gc, st, args = self._card(33.0, 0.42, 0.26)
        args["set_fan"] = lambda f: f is None                        # the driver refuses every speed
        prof = gc.measure(**args)
        self.assertIs(prof["fans"], False)
        self.assertLess(prof["power_w"], 170.0)
        self.assertFalse(st["load"])


class MaxMode(unittest.TestCase):
    def test_full_power_never_paces(self):
        class PC:
            top, limit, via = 170000, 136000.0, "directly"
            seen = []

            def set(self, mw):
                PC.seen.append(mw)
        with mock.patch.object(thermal._PowerControl, "try_make", staticmethod(lambda *a: PC())), \
                mock.patch.dict(os.environ, {"APSI_GPU_TEMP": "max"}):
            temps = iter([86.0] * 1000)
            g = thermal._max_governor(None, None, 0, lambda: next(temps))
            self.assertEqual(PC.seen, [170000])
            g.retarget(79.0)
            clock = [0.0]
            with mock.patch.object(thermal.time, "monotonic", lambda: clock[0]):
                for _ in range(200):
                    clock[0] += 0.1
                    g.tick()
            self.assertEqual(g.fraction, 0.0)
            self.assertEqual(g.paced, 0.0)
