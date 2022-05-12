import enum
import queue
import threading
import numpy as np
from neuwon import Model

class Message(enum.Enum):
    INSTANCE    = enum.auto()
    COMPONENT   = enum.auto()
    SET_TIME    = enum.auto()
    DURATION    = enum.auto()
    RUN         = enum.auto()
    PAUSE       = enum.auto()

class ModelRunner(threading.Thread):
    def __init__(self):
        super().__init__(name='ModelRunner')
        # The control_queue contains pairs of (Message, payload) where the
        # payload type depends on the type of message.
        self.control_queue = queue.Queue()
        self.results_queue = queue.Queue(maxsize=10)
        self._instance  = None # Instance of neuwon.Model().
        self._active    = False # Is the model currently running or is it stopped?
        self._component = None # If None then results_queue is not used.
        self._duration  = None # Integer number of time_steps.
        self._quit      = False
        self.start()

    def run(self):
        while not self._quit:
            self._update_control()
            self._update_model()

    def quit(self):
        self._quit = True

    def _update_control(self):
        while True:
            block = (not self._active)
            try:
                # Block only for a short period of time, so that the quit method works.
                message, payload = self.control_queue.get(block=block, timeout=3)
            except queue.Empty:
                break

            if message == Message.INSTANCE:
                assert not self._active
                assert isinstance(payload, Model)
                self._instance = payload

            elif message == Message.COMPONENT:
                if payload is None:
                    self._component = None
                else:
                    self._component = payload
                    self._instance.get_component(self._component) # assert component exists.

            elif message == Message.SET_TIME:
                assert not self._active
                clock = self._instance.get_clock()
                time = float(payload)
                if payload != clock.get_time():
                    clock.set_time(payload)

            elif message == Message.DURATION:
                assert not self._active
                duration = float(payload)
                assert duration >= 0
                if duration == np.inf:
                    self._duration = duration
                else:
                    clock = self._instance.get_clock()
                    self._duration = round(duration / clock.get_tick_period())

            elif message == Message.RUN:
                assert self._instance   is not None
                assert self._duration   is not None
                self._active = True

            elif message == Message.PAUSE:
                self._active = False

            else:
                raise NotImplementedError(message)

    def _update_model(self):
        if not self._active:
            return
        elif self._duration <= 0:
            self._active = False
            return
        self._instance.advance()
        self._duration -= 1
        if self._component is not None:
            render_data = self._instance.get_database().get_data(self._component)
            render_data = np.array(render_data, copy=True)
            self.results_queue.put(render_data)

    def is_running(self):
        return self._active

    def get_time(self):
        if self._instance is None:
            return np.nan
        return self._instance.get_clock().get_time()

    def get_duration(self):
        if self._instance is None or self._duration is None:
            return np.nan
        return self._duration * self._instance.get_clock().get_tick_period()
