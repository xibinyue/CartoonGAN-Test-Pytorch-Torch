#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import logging
import logging.config
import os
from logging import Logger
from logging.handlers import TimedRotatingFileHandler


class MyLogger(object):
    def __init__(self, logger_name, logger_file, is_print=False, level=logging.DEBUG):
        self.loggerName = logger_name
        self.loggerFile = logger_file
        self.isPrint = is_print
        self.level = level

    def Logger(self):
        logger = logging.getLogger(self.loggerName)
        logger.setLevel(self.level)
        th = TimedRotatingFileHandler(self.loggerFile, when="midnight")
        fh = logging.FileHandler(self.loggerFile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        th.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(th)
        if self.isPrint:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger
