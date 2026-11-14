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
            self.assertGreaterEqual(min(full), 78.0, (tau, amb))
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
    """Faster fans near the ceiling (the driver's curve below it), back to the driver's curve at the end."""

    def test_fan_curve_and_restore(self):
        calls = []
        fc = thermal._FanControl()
        fc.top, fc.percent = 85, None
        fc._set = lambda pct: calls.append(pct) or True
        fc.step(65.0, 79.0)                              # cool: the driver's own curve, nothing set
        self.assertEqual(calls, [])
        fc.step(74.0, 79.0)                              # ceiling - 5: on the way up
        fc.step(77.5, 79.0)                              # past ceiling - 2: the top
        self.assertEqual(calls[-1], 85)
        self.assertTrue(50 <= calls[0] < 85)
        fc.auto()
        self.assertIsNone(calls[-1])

    def test_more_cooling_more_power(self):
        """With the fans at the top near the ceiling the simulated card keeps more power at 78-79 C."""
        def run(with_fan):
            clock, st = [0.0], {"T": 45.0, "limit": 136000.0, "fan": 0.0}

            def power():
                return min(st["limit"] / 1000.0, 175.0)

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
            if with_fan:
                fc = thermal._FanControl()
                fc.top, fc.percent = 85, None
                fc._set = lambda pct: st.__setitem__("fan", 0.0 if pct is None else (pct - 50) / 35.0) or True
                g.fan = fc
            watts = []
            with mock.patch.object(thermal.time, "monotonic", lambda: clock[0]):
                while clock[0] < 300.0:
                    r = 0.30 * (1.0 - 0.15 * st["fan"])      # faster fans: less temperature rise per watt
                    st["T"] += (32.0 + r * power() - st["T"]) * min(1.0, 0.05 / 20.0)
                    clock[0] += 0.05
                    g.tick()
                    if clock[0] > 200:
                        watts.append(power())
                        self.assertLess(read(), 80.0)
            return sum(watts) / len(watts)
        self.assertGreater(run(True), run(False) * 1.08)
