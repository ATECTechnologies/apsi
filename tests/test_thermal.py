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


if __name__ == "__main__":
    unittest.main()
