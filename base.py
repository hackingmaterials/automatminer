"""
Base classes, mixins, and other inheritables.
"""

__authors__ = ["Alex Dunn <ardunn@lbl.gov>"]


class Loggable:
    def _log(self, lvl, msg):
        """
        Convenience method for logging.

        Args:
            lvl (str): Level of the log message, either "info", "warn", or "debug"
            msg (str): The message for the logger.

        Returns:
            None
        """
        if not hasattr(self, "logger"):
            raise AttributeError("Loggable object has no logger object!")

        if self.logger is not None:
            if lvl == "warn":
                self.logger.warning(msg)
            elif lvl == "info":
                self.logger.info(msg)
            elif lvl == "debug":
                self.logger.debug(msg)