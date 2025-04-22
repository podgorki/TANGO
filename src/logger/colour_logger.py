
import colorlog
# logger = colorlog.getLogger(__name__)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))

logger = colorlog.getLogger("Localizer")
logger.addHandler(handler)
logger.setLevel(colorlog.INFO)