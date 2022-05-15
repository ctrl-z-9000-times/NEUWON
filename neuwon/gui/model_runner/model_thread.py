from neuwon import Model
from threading import Thread
from time import sleep
import collections
import enum
import numpy as np
import queue

class Message(enum.Enum):
    INSTANCE    = enum.auto()
    COMPONENT   = enum.auto()
    HEADLESS    = enum.auto()
    SET_TIME    = enum.auto()
    DURATION    = enum.auto()
    RUN         = enum.auto()
    PAUSE       = enum.auto()
    QUIT        = enum.auto()

Result = collections.namedtuple('Result', ['timestamp', 'remaining', 'data'])

class ModelThread(Thread):
    """
    Run the model in its own thread so that it does not block the GUI while it's
    running the simulation.
    """
    def __init__(self):
        super().__init__(name='ModelThread')
        # The control_queue contains pairs of (Message, payload) where the
        # payload type depends on the type of message.
        self.control_queue = queue.Queue()
        self.results_queue = queue.Queue(10) # Do not run too far ahead of the GUI.
        self._instance  = None  # Instance of neuwon.Model().
        self._active    = False # Is the model currently running or is it stopped?
        self._component = None  # Database component to send to GUI after each simulation tick.
        self._headless  = False # Disconnected from graphical output?
        self._duration  = None  # Integer number of time_steps.
        self._quit      = False
        self.exception  = None
        self.start()

    def run(self):
        while not self._quit:
            self._update_control()
            self._update_model()

    def _update_control(self):
        while True:
            block = (not self._active)
            try:
                # Do not block forever, due to issues in the underlying lock on some platforms.
                message = self.control_queue.get(block=block, timeout=9999)
            except queue.Empty:
                break
            if isinstance(message, collections.abc.Iterable):
                message, payload = message

            if message == Message.INSTANCE:
                assert not self._active
                assert isinstance(payload, Model)
                self._instance = payload

            elif message == Message.COMPONENT:
                self._component = payload
                self._instance.get_database().get_component(self._component) # assert component exists.

            elif message == Message.HEADLESS:
                self._headless = bool(payload)

            elif message == Message.SET_TIME:
                assert not self._active
                clock = self._instance.get_clock()
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
                assert self._instance is not None
                assert self._duration is not None
                self._active = True

            elif message == Message.PAUSE:
                self._active = False

            elif message == Message.QUIT:
                self._quit = True
                return

            else:
                raise NotImplementedError(message)

    def _update_model(self):
        if not self._active:
            return
        elif self._duration <= 0:
            self._active = False
            self.results_queue.put(Message.PAUSE)
            return
        # 
        try:
            self._instance.advance()
        except Exception as x:
            self.exception = x
            raise x
        self._duration -= 1
        # Gather the results and output them.
        clock     = self._instance.get_clock()
        timestamp = clock.get_time()
        remaining = self._duration * clock.get_tick_period()
        if self._headless:
            render_data = None
        else:
            render_data = self._instance.get_database().get_data(self._component)
            render_data = np.array(render_data, copy=True)
        self.results_queue.put(Result(timestamp, remaining, render_data))
