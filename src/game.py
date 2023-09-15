import numpy as np

from model import*

class Maze:
    def __init__(self,filename_trail):
        points = []
        self.edges = []
        self.destination = []
        with open(filename_trail[0], mode='r') as file:
            point_buffer = []
            for index, line in enumerate(file):
                point_buffer = []
                line = list(map(int, line.strip().split(',')))
                if index == 0:
                    continue
                if index == 1 or index == 2:
                    self.destination.append(line[0])
                    self.destination.append(line[1])
                else:
                    point_buffer.append(line[0])
                    point_buffer.append(line[1])
                    points.append(point_buffer)
        self.edges = []
        for i in range(len(points)-1):
            edge = points[i]+points[i+1]
            self.edges.append(edge)
        self.edges = np.array(self.edges)
        self.destination = np.array(self.destination)

class Car(Maze):
    def __init__(self,filename_trail):
        super().__init__(filename_trail)
        with open(filename_trail[0],mode='r') as file:
            for index, line in enumerate(file):
                if index == 0:
                    line = list(map(int, line.strip().split(',')))
                    self.x = line[0]
                    self.y = line[1]
                    self.angle = line[2]
        self.radius = 3
        self.is_success = False
        self.is_failed = False

    @staticmethod
    def euclidean_dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def quadrant_dir(angle):
        if angle >= 360:
            angle %= 360
        if angle < 0:
            angle *= (-1)
            angle %= 360
            angle *= (-1)
            angle += 360
            angle %= 360
        if 0 < angle < 90:
            return 1
        elif 90 < angle < 180:
            return 2
        elif 180 < angle < 270:
            return 3
        elif 270 < angle < 360:
            return 4
        elif angle == 0:
            return "right"
        elif angle == 90:
            return "up"
        elif angle == 180:
            return "left"
        elif angle == 270:
            return "down"

    def probe(self,is6d):
        sensor_data = []
        if is6d:
            sensor_data.append(self.x)
            sensor_data.append(self.y)
        #[front, right, left]
        sensors = [0, -45, 45]
        for sensor in sensors:
            angle = self.angle + sensor
            dist_arr = []
            for edge in self.edges:
                a = np.array([
                    [edge[3]-edge[1],edge[0]-edge[2]],
                    [np.asscalar(-1*np.tan(angle*np.pi/180)),1]])
                ax = np.array([
                    [edge[0]*edge[3]-edge[1]*edge[2], edge[0] - edge[2]],
                    [np.asscalar(-1 * np.tan(angle * np.pi / 180)*self.x + self.y), 1]])
                ay = np.array([
                    [edge[3] - edge[1],edge[0]*edge[3]-edge[1]*edge[2]],
                    [np.asscalar(-1*np.tan(angle*np.pi/180)), np.asscalar(-1 * np.tan(angle * np.pi / 180)*self.x + self.y)]
                ])
                det_a = np.linalg.det(a)
                det_ax = np.linalg.det(ax)
                det_ay = np.linalg.det(ay)
                if det_a == 0:
                    continue
                x = det_ax/det_a
                y = det_ay/det_a
                #avoid float bias
                x = np.round(x,2)
                y = np.round(y,2)
                #check x y rationality wrt. edge
                ab_vec = np.array([edge[2]-edge[0],edge[3]-edge[1]])
                ac_vec = np.array([x-edge[0],y-edge[1]])
                if not (np.dot(ab_vec, ab_vec) >= np.dot(ab_vec, ac_vec) >= 0):
                    # print("not on wall")
                    continue
                #check x y rationality wrt. sensor
                quadrant = self.quadrant_dir(angle)
                if quadrant == 1:
                    if not(x > self.x and y > self.y):
                        continue
                if quadrant == 2:
                    if not(x < self.x and y > self.y):
                        continue
                if quadrant == 3:
                    if not(x < self.x and y < self.y):
                        continue
                if quadrant == 4:
                    if not(x > self.x and y < self.y):
                        continue
                if quadrant == "right":
                    if not(x > self.x):
                        continue
                if quadrant == "left":
                    if not(x < self.x):
                        continue
                if quadrant == "up":
                    if not(y > self.y):
                        continue
                if quadrant == "down":
                    if not(y < self.y):
                        continue
                # print("---------------------------------")
                # print("x: ", x, " y: ", y)
                # print("quadrant: ", quadrant)
                # print("---------------------------------")
                dist_arr.append(self.euclidean_dist(np.array([self.x,self.y]),np.array([x,y])))
            # print("dist arr: ",dist_arr)
            dist_arr = np.array(dist_arr)
            dist = np.min(dist_arr)
            sensor_data.append(dist)
            # print("sensor data: ",sensor_data)
        sensor_data = np.array(sensor_data)
        return sensor_data

    def point_to_line_dist(self,edge):
        point_line_vec = np.array([(edge[0]-self.x).item(),(edge[1]-self.y).item()])
        line_vec = np.array([edge[2]-edge[0],edge[3]-edge[1]])
        point_line_vec_norm = np.linalg.norm(point_line_vec)
        line_vec_norm = np.linalg.norm(line_vec)
        cos_theta = np.dot(point_line_vec,line_vec)/(point_line_vec_norm*line_vec_norm)
        theta = np.arccos(cos_theta)
        return point_line_vec_norm*np.sin(theta)

    def termination_check(self):
        for edge in self.edges:
            curr_pos = np.array([self.x,self.y])
            endpoint_1 = np.array([edge[0],edge[1]])
            endpoint_2 = np.array([edge[2],edge[3]])
            curr_pos_endpoint_1_dist = self.euclidean_dist(curr_pos, endpoint_1)
            curr_pos_endpoint_2_dist = self.euclidean_dist(curr_pos,endpoint_2)
            curr_pos_to_edge_dist = self.point_to_line_dist(edge)
            edge_length = self.euclidean_dist(endpoint_1, endpoint_2)
            endpoint_touch = True if curr_pos_endpoint_1_dist < self.radius or curr_pos_endpoint_2_dist < self.radius else False
            in_edge_touch = (curr_pos_to_edge_dist<self.radius and (curr_pos_endpoint_1_dist < edge_length and curr_pos_endpoint_2_dist < edge_length))
            if (self.destination[0] < self.x < self.destination[2] ) and (self.destination[3] <self.y<self.destination[1]):
                self.is_success = True if not self.is_failed else False
            if endpoint_touch or in_edge_touch:
                self.is_failed = True if not self.is_success else False

    def move(self,predicted_control):
        if predicted_control > 40:
            predicted_control = 40
        if predicted_control < -40:
            predicted_control = -40
        curr_angle_radian = self.angle* np.pi / 180
        predicted_control_radian = predicted_control * np.pi / 180
        self.x += (np.cos(curr_angle_radian + predicted_control_radian) + np.sin(predicted_control_radian)*np.sin(curr_angle_radian)).item()
        self.y += (np.sin(curr_angle_radian + predicted_control_radian) - np.sin(predicted_control_radian)*np.cos(curr_angle_radian)).item()
        self.angle -= (np.arcsin(2*np.sin(predicted_control_radian)/(self.radius*2))*180/np.pi).item()
        if self.angle > 270:
            self.angle = 270
        if self.angle < -90:
            self.angle = -90
        self.termination_check()
