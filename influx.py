from influxdb import InfluxDBClient
import datetime
import matplotlib.pyplot as plt
import json

class InfluxDBHelper:
    def __init__(self, host, port, username, password, database):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.client = InfluxDBClient(host=self.host, port=self.port, username=self.username, password=self.password, database=self.database)

    def write_data(self, measurement, fields, tags=None):
        data = [
            {
                "measurement": measurement,
                "tags": tags,
                "fields": fields
            }
        ]
        self.client.write_points(data)

    def query_data(self, query):
        result = self.client.query(query)
        return result.raw

    def close_connection(self):
        self.client.close()


