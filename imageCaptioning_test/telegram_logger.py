import requests
from logging import Handler, Formatter
import logging
import datetime

class RequestsHandler(Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': log_entry,
            'parse_mode': 'HTML'
        }
        return requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=TELEGRAM_TOKEN),
                             data=payload).content

class LogstashFormatter(Formatter):
    def __init__(self):
        super(LogstashFormatter, self).__init__()

    def format(self, record):
        t = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        return "<i>{datetime}</i><pre>\n{message}</pre>".format(message=record.msg, datetime=t)

#I use a simple telegram bot logger to get notifications about the
# training process.
#
TELEGRAM_TOKEN = '*'   #Telegram bot token
TELEGRAM_CHAT_ID = '*'              #My telegram user id
TELEGRAM_LOG_ACTIVATE = True                #A variable to turn on/off notifications

#Setup the telegram logger. This way, everytime we want to send something to the smartphone,
# we only need to call for logger.error("<message>")
logger = logging.getLogger('telegram-logger')
logger.setLevel(logging.WARNING)
handler = RequestsHandler()
formatter = LogstashFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

def telegramSendMessage(message):
    """
        Envia uma mensagem de log para o telegram
    """
    if TELEGRAM_LOG_ACTIVATE:
        logger.error(message)