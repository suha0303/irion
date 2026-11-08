"""Use pymavlink's Python indexer without changing the shared
installation/env.
"""


def open_bin(path):
    from pymavlink import DFReader

    class PythonIndexedBIN(DFReader.DFReader_binary):
        def init_arrays_fast(self, progress_callback=None):
            # The parent chooses this method from its environment. Never enter
            # the
            # optional native indexer, even when PYMAVLINK_FAST_INDEX=1 is
            # inherited.
            return self.init_arrays(progress_callback=progress_callback)

    return PythonIndexedBIN(str(path))
