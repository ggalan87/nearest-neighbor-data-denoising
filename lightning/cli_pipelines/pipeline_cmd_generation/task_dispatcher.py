import os
import select
from pathlib import Path
import subprocess
import threading
from inotify_simple import INotify, masks, flags


class InotifyThread(threading.Thread):
    def __init__(self, path):
        self._path = path
        self._path.mkdir(exist_ok=True)

        # Initialize the parent class
        threading.Thread.__init__(self)

        # Create an inotify object
        self._inotify = INotify()

        # Create a pipe
        self._read_fd, write_fd = os.pipe()
        self._write = os.fdopen(write_fd, "wb")

    def run(self):
        # Watch the current directory
        self._inotify.add_watch(self._path, masks.ALL_EVENTS)
        self._inotify.add_watch(self._path, flags.CREATE)

        while True:
            # Wait for inotify events or a write in the pipe
            rlist, _, _ = select.select(
                [self._inotify.fileno(), self._read_fd], [], []
            )

            # Print all inotify events
            if self._inotify.fileno() in rlist:
                for event in self._inotify.read(timeout=0):
                    event_flags = flags.from_mask(event.mask)

                    # special case to stop the thread
                    if event.name == 'stop':
                        (self._path / event.name).unlink(missing_ok=True)
                        self.stop()
                    elif event.name[-3:] == '.sh':
                        script_path = self._path / event.name

                        working_dir = script_path.resolve().parent
                        print(f'Dispatching file {event.name} in {working_dir}')

                        subprocess.run(str(script_path), shell=True, cwd=working_dir)

                        if script_path.is_symlink():
                            script_path.unlink()

                    # event_flags_names = [f.name for f in event_flags]
                    # print(f"{event} {event_flags_names}")

            # Close everything properly if requested
            if self._read_fd in rlist:
                os.close(self._read_fd)
                self._inotify.close()
                return

    def stop(self):
        # Request for stop by writing in the pipe
        if not self._write.closed:
            self._write.write(b"\x00")
            self._write.close()


experiments_path = Path('/tmp/experiments')
task_dispatcher = InotifyThread(experiments_path)
task_dispatcher.run()
