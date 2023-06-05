# -*- coding: utf-8 -*-
import json
import uuid
import random

from .basisRequestHandler import BasisRequestHandler


class GameStartHandler(BasisRequestHandler):
    def initialize(self, datamgr):
        self.datamgr = datamgr

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def options(self):
        self.set_status(204)
        self.finish()

    def __generate_unique_identifier(self):
        return uuid.uuid4().hex

    def post(self):
        # Generate new uuid
        user_id = self.__generate_unique_identifier()

        # Randomly assign user to one of the groups
        # 0 = control, no explanation
        # 1 = group 1: only upwards
        # 2 = group 2: only downwards
        # 3 = group 3: mixed
        #expGroup = 0
        expGroup = 1
        #expGroup = 2
        #expGroup = 3
        print(expGroup)
        flips = [random.randint(0,3) for i in range(160)]
        zeros = flips.count(0)
        ones = flips.count(1)
        twos = flips.count(2)
        threes = flips.count(3)
        print(zeros, ones, twos, threes)

        # Add it to the database
        if self.datamgr.add_new_user(user_id, expGroup) is False:
            self.send_custom_error(500, "Internal server error")
        else:
            # Send it back to the client
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"userId": user_id, "expGroup": expGroup}))
            self.finish()
