import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from core.gesture_gate import classify_number, NumberGestureSelector, ViewGestureGate
from core.tool_catalog import PALETTE, palette_bgr
from core.viewport import Viewport
from core.state import state, AppState
from core.pipeline import VideoPipeline


def number_hand(number):
    points = [(0, 0)] * 21
    points[0] = (0, 120)
    points[1:5] = [(-40, 95), (-60, 70), (-25, 65), (-5, 70)]
    patterns = {1: [1, 0, 0, 0], 2: [1, 1, 0, 0], 3: [1, 1, 1, 0],
                4: [1, 1, 1, 1], 5: [1, 1, 1, 1], 6: [0, 0, 0, 1], 7: [0, 0, 0, 0], 0: [0, 0, 0, 0]}
    for mcp, (x, y), extended in zip((5, 9, 13, 17), ((-30, 40), (0, 30), (30, 40), (60, 55)), patterns[number]):
        points[mcp] = (x, y)
        points[mcp + 1:mcp + 4] = [(x, y - 35), (x, y - 65), (x, y - 90)] if extended else [(x, y - 25), (x + 10, y - 10), (x + 5, y + 15)]
    if number in (5, 6):
        points[1:5] = [(-40, 95), (-65, 75), (-90, 55), (-115, 35)]
    if number == 7:
        points[4], points[8], points[12] = (-10, -10), (0, -15), (10, -10)
    points = [(x + 300, y + 180) for x, y in points]
    return {'landmarks': points, 'index_pos': points[8], 'fingers': [bool(v) for v in patterns[number]]}


def right_hand(pos=(200, 200), gap=1.0):
    hand = number_hand(1)
    points = hand['landmarks']
    points[5], points[17] = (200, 300), (300, 300)
    points[8] = pos
    points[4] = (pos[0] + gap * 100, pos[1])
    points[12] = (points[4][0], pos[1] - 150)
    hand['index_pos'] = pos
    return hand


class GestureTests(unittest.TestCase):
    def test_numbers_and_rotated_mirrored_scaled_hands(self):
        for number in range(1, 8):
            hand = number_hand(number)
            self.assertEqual(classify_number(hand), number)
            for mirror in (-1, 1):
                changed = copy.deepcopy(hand)
                changed['landmarks'] = [(mirror * 1.5 * (x * math.cos(.6) - y * math.sin(.6)) + 30,
                                         1.5 * (x * math.sin(.6) + y * math.cos(.6)) + 40) for x, y in hand['landmarks']]
                self.assertEqual(classify_number(changed), number)
        self.assertIsNone(classify_number(number_hand(0)))
        self.assertIsNone(classify_number(None))

    def test_dwell_latch_and_clear_confirmation(self):
        gate = NumberGestureSelector()
        self.assertIsNone(gate.update_number(1, 0))
        self.assertIsNone(gate.update_number(1, .49))
        self.assertEqual(gate.update_number(1, .51), 1)
        self.assertIsNone(gate.update_number(1, 3))
        gate.update_number(None, 3.1)
        gate.update_number(None, 3.4)
        gate.update_number(3, 3.5)
        self.assertIsNone(gate.update_number(3, 4.1))
        self.assertEqual(gate.update_number(3, 4.71), 3)
        self.assertIsNone(gate.update_number(3, 8))

    def test_transient_number_does_not_fire(self):
        gate = NumberGestureSelector()
        for number, now in ((1, 0), (2, .1), (3, .2), (4, .3), (5, .4)):
            self.assertIsNone(gate.update_number(number, now))
        self.assertEqual(gate.update_number(5, .91), 5)

    def test_view_ambiguous_pinch_and_release(self):
        gate = ViewGestureGate()
        hand = right_hand((200, 200), .1)
        for _ in range(3):
            self.assertIsNone(gate.update(hand))
        gate.update(None)
        hand['landmarks'][12] = (215, 200)
        for _ in range(3):
            self.assertIsNone(gate.update(hand))
        hand['landmarks'][12] = (300, 100)
        self.assertIsNone(gate.update(hand))
        self.assertEqual(gate.update(hand), 'PAN')
        hand['landmarks'][12] = (215, 200)
        self.assertIsNone(gate.update(hand))
        hand['landmarks'][12] = (300, 100)
        self.assertIsNone(gate.update(hand))
        gate.update(None)
        self.assertIsNone(gate.update(hand))
        self.assertEqual(gate.update(hand), 'PAN')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.old_state = copy.deepcopy(state.__dict__)
        state.__dict__.update(AppState().__dict__)
        with patch('core.pipeline.HandTracker'):
            self.pipeline = VideoPipeline()
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def tearDown(self):
        state.__dict__.clear()
        state.__dict__.update(self.old_state)

    def step(self, hand=None):
        return self.pipeline.step(self.frame.copy(), None, hand)

    def test_palette_all_colors_and_no_ink(self):
        for i, (x, y, w, h) in enumerate(Viewport.palette_rects()):
            before = self.pipeline.canvas_engine.canvas.copy()
            self.pipeline.select_tool('COLOR')
            # Opening the right hand arms the newly selected tool.
            self.step(right_hand((x + w // 2, y + h // 2), 1))
            self.step(right_hand((x + w // 2, y + h // 2), .1))
            self.step(right_hand((x + w // 2, y + h // 2), .1))
            self.assertEqual(state.color, palette_bgr(i))
            self.assertEqual(state.tool, 'PAINT')
            self.assertTrue(self.pipeline.stroke_locked)
            self.step(right_hand((400, 400), .1))
            np.testing.assert_array_equal(self.pipeline.canvas_engine.canvas, before)
            self.step(None)
            self.pipeline.select_tool('COLOR')
            self.pipeline.click_local((x + 1, y + 1))
            self.assertEqual(state.color, palette_bgr(i))
            self.assertEqual(state.tool, 'PAINT')
        self.assertIsNone(Viewport.palette_hit((0, 0)))

    def test_draw_erase_clear_and_preview_isolation(self):
        state.enable_shape = False
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((220, 200), .1))
        self.step(None)
        world = self.pipeline.viewport.to_world((200, 200))
        self.assertTrue((self.pipeline.canvas_engine.canvas[world[1], world[0]] == 0).all())
        self.pipeline.select_tool('ERASE')
        self.step(None)
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((200, 200), .1))
        self.assertTrue((self.pipeline.canvas_engine.canvas[world[1], world[0]] == 255).all())
        self.pipeline.select_tool('COLOR')
        before = self.pipeline.canvas_engine.canvas.copy()
        self.step(None)
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)
        self.pipeline.select_tool('CLEAR')
        self.assertTrue((self.pipeline.canvas_engine.canvas == 255).all())

    def test_combined_view_zoom_pan_and_reentry(self):
        def zoom_hand(y):
            hand = right_hand((400, y), 1)
            hand['landmarks'][4] = (200, y)
            hand['landmarks'][12] = (210, y)
            return hand
        self.pipeline.select_tool('VIEW')
        before = self.pipeline.canvas_engine.canvas.copy()
        self.step(None)
        self.step(zoom_hand(300))
        self.step(zoom_hand(300))
        self.step(zoom_hand(200))
        self.assertGreater(self.pipeline.viewport.zoom, 2)
        self.step(zoom_hand(450))
        self.assertEqual(self.pipeline.viewport.zoom, 1)
        self.step(zoom_hand(0))
        self.assertEqual(self.pipeline.viewport.zoom, 8)
        self.pipeline.set_zoom(3)
        self.step(zoom_hand(200))
        self.assertEqual(self.pipeline.viewport.zoom, 3)
        self.step(None)
        self.step(zoom_hand(400))
        self.step(zoom_hand(400))
        self.assertEqual(self.pipeline.viewport.zoom, 3)
        # Switching pinch types without release cannot start a pan.
        center = self.pipeline.viewport.cx
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((240, 200), .1))
        self.assertEqual(self.pipeline.viewport.cx, center)
        self.step(None)
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((240, 200), .1))
        self.assertLess(self.pipeline.viewport.cx, center)
        self.assertEqual(state.tool, 'VIEW')
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)

    def test_switch_requires_actual_release_and_pan_does_not_draw(self):
        self.pipeline.select_tool('PAN')
        before = self.pipeline.canvas_engine.canvas.copy()
        for _ in range(3):
            self.step(right_hand((200, 200), .1))
        self.assertIsNone(self.pipeline.drag)
        self.step(right_hand((200, 200), 1))
        self.step(right_hand((200, 200), .1))
        self.step(right_hand((200, 200), .1))
        center = self.pipeline.viewport.cx
        self.step(right_hand((240, 200), .1))
        self.assertLess(self.pipeline.viewport.cx, center)
        self.step(None)
        self.assertIsNone(self.pipeline.drag)
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)

    def test_filter_modes_reset_and_shapes_still_commit(self):
        for mode in ('none', 'fixed', 'adaptive'):
            state.filter_mode = mode
            state.enable_shape = False
            self.pipeline.select_tool('PAINT')
            self.step(None)
            self.step(right_hand((200, 200), .1))
            self.step(right_hand((200, 200), .1))
            hand = right_hand((230, 200), .1)
            self.step(hand)
            if mode == 'none':
                self.assertEqual(hand['filtered_index_pos'], (230, 200))
            else:
                self.assertLess(hand['filtered_index_pos'][0], 230)
            self.step(None)
            self.assertIsNone(self.pipeline.filter_processor.prev_pos)
            self.assertIsNone(self.pipeline.canvas_engine.prev_pos)
        state.enable_shape = True
        for vertices in (((200, 200), (400, 200), (400, 400), (200, 400)),
                         ((200, 400), (300, 200), (400, 400))):
            points = []
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                points.extend(tuple(round(a + (b - a) * t) for a, b in zip(start, end))
                              for t in np.linspace(0, 1, 12))
            self.pipeline.canvas_engine.clear()
            for point in points:
                self.pipeline.canvas_engine.draw_or_erase(point, 'PAINT')
            with patch.object(self.pipeline.canvas_engine, 'replace_last_stroke',
                              wraps=self.pipeline.canvas_engine.replace_last_stroke) as replace:
                self.pipeline.finish_stroke()
                replace.assert_called_once()
            self.assertTrue(np.any(self.pipeline.canvas_engine.canvas < 255))

    def test_neural_generation_cancel_and_commit(self):
        from unittest.mock import Mock
        painter = self.pipeline.portrait
        before = self.pipeline.canvas_engine.canvas.copy()
        job = Mock()
        job.poll.return_value = None
        job.percent = 25
        painter.job = job
        painter.pending_alpha = np.ones((40, 40), np.float32)
        self.assertTrue(painter.busy)
        self.assertIn('25%', painter.poll_generation(self.pipeline.canvas_engine.canvas))
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)
        self.pipeline.select_tool('PAINT')
        job.cancel.assert_called_once()
        self.assertFalse(painter.busy)
        job = Mock()
        job.poll.return_value = np.zeros((40, 40, 3), np.uint8)
        painter.job = job
        painter.pending_alpha = np.ones((40, 40), np.float32)
        painter.poll_generation(self.pipeline.canvas_engine.canvas, now=0)
        self.assertIsNone(painter.job)
        self.assertTrue(painter.active)
        painter.advance(self.pipeline.canvas_engine.canvas, now=.15)
        self.pipeline.cancel_portrait()
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)

    def test_neural_styles_use_entire_frame_without_head_extraction(self):
        from unittest.mock import Mock
        painter = self.pipeline.portrait
        frame = np.full((480, 640, 3), (30, 80, 120), np.uint8)
        frame[0, 0] = (1, 2, 3)
        with patch.object(painter, 'detector') as detector, patch.object(painter.head, 'extract') as extract:
            for style in ('van_gogh', 'picasso'):
                with patch('modules.neural_portrait.NeuralPortraitJob') as factory:
                    painter.start(frame, self.pipeline.canvas_engine.canvas, (0, 0, 1280, 960), style=style)
                    sent = factory.call_args.args[0]
                    np.testing.assert_array_equal(sent, frame)
                    self.assertIsNot(sent, frame)
                    self.assertIsNone(painter.pending_alpha)
                    painter.cancel(self.pipeline.canvas_engine.canvas)
            detector.empty.assert_not_called()
            extract.assert_not_called()
        before = self.pipeline.canvas_engine.canvas.copy()
        painter.job = Mock()
        painter.job.poll.return_value = frame
        painter.pending_alpha = None
        painter.poll_generation(self.pipeline.canvas_engine.canvas, now=0)
        self.assertEqual((painter.x, painter.y), (0, 0))
        self.assertEqual(painter.target.shape, before.shape)
        self.assertTrue((painter.alpha == 1).all())
        painter.advance(self.pipeline.canvas_engine.canvas, now=.15)
        painter.cancel(self.pipeline.canvas_engine.canvas)
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)

    def test_neural_failure_releases_job(self):
        from unittest.mock import Mock
        job = Mock()
        job.poll.side_effect = ValueError('测试生成失败')
        self.pipeline.portrait.job = job
        state.tool = 'PORTRAIT'
        self.step(None)
        self.assertIn('测试生成失败', self.pipeline.message)
        self.assertFalse(self.pipeline.portrait.busy)
        job.cancel.assert_called_once()

    def test_portrait_cancels_on_tool_change(self):
        painter = self.pipeline.portrait
        before = self.pipeline.canvas_engine.canvas.copy()
        painter.begin_image(np.zeros((40, 40, 3), np.uint8), self.pipeline.canvas_engine.canvas, (0, 0, 100, 100), now=0)
        painter.advance(self.pipeline.canvas_engine.canvas, now=.15)
        self.pipeline.select_tool('COLOR')
        self.assertFalse(painter.active)
        np.testing.assert_array_equal(before, self.pipeline.canvas_engine.canvas)


if __name__ == '__main__':
    unittest.main()
