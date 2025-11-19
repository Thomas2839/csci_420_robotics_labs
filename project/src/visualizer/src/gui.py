import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy
from matplotlib.patches import Polygon


class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads, env, map_size):
        self.quads = quads
        self.world = env
        self.fig = plt.figure()
        plt.ion()
        self.fig.canvas.draw()
        plt.show(block=False)
        # self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax = plt.axes()
        self.map_size = map_size
        self.init_map_size()
        self.crashed = False
        self.line = None
        self.lidar_line = None
        self.polygons = {}
        # Used to check if redrawing is required
        self.last_draw_path = np.zeros(10)
        self.collections = []
        self.last_draw_height = -1
        self.init_plot()

    def init_map_size(self):
        x = self.map_size[0]
        y = self.map_size[1]
        self.ax.set_xlim([-1 * x, x])
        self.ax.set_xlabel('X')
        self.ax.set_ylim([-1 * y, y])
        self.ax.set_ylabel('Y')
        self.ax.set_title('Quadcopter Simulation')

    def rotation_matrix(self,   angle):
        # ct = math.cos(angles[0])
        # cp = math.cos(angles[1])
        # cg = math.cos(angles[2])
        # st = math.sin(angles[0])
        # sp = math.sin(angles[1])
        # sg = math.sin(angles[2])
        # R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        # R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        # R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        # R = np.dot(R_z, np.dot( R_y, R_x ))
        ct = math.cos(angle)
        st = math.sin(angle)
        R = np.array([[ct, -st],[st, ct]])
        return R

    def draw_path(self, path):
        self.line.set_xdata(path[:,0])
        self.line.set_ydata(path[:,1])
        #self.line.set_3d_properties(np.full(len(path[:,1]), 5))
        self.last_draw_path = copy.deepcopy(path)

    def draw_polygon(self, loc_x, loc_y, color):
        loc_x -= 0.5
        loc_y -= 0.5
        loc = (loc_x, loc_y)
        if loc in self.polygons:
            p = self.polygons[loc]
            p.set_facecolor(color)
        else:
            x_obs = [loc_x, loc_x + 1, loc_x + 1, loc_x]
            y_obs = [loc_y, loc_y, loc_y + 1, loc_y + 1]
            verts = [list(x) for x in zip(x_obs, y_obs)]
            p = Polygon(verts, facecolor=color)
            self.ax.add_patch(p)
            self.polygons[loc] = p

    def draw_world(self, height):
        for collection in self.collections:
            collection.remove()
        self.collections = []
        for obs in self.world["environment"]:
            obs_x = obs[0]
            obs_y = obs[1]
            color = 1 - obs[2] / 100.0  # as given, larger value means filled, so that needs to be black
            color = (color, color, color)  # expand to rgb
            self.draw_polygon(obs_x, obs_y, color)
        for door in self.world["doors"]:
            door_x = door[0]
            door_y = door[1]
            state = door[2]
            color = 'cyan' if state == 'closed' else 'green'
            self.draw_polygon(door_x, door_y, color)

        if 'goal' in self.world:
            goal_x = self.world['goal'][0]
            goal_y = self.world['goal'][1]
            self.draw_polygon(goal_x, goal_y, 'magenta')
            self.last_draw_height = height
        try:
            if 'lidar' in self.world and len(self.world['lidar']) > 0:
                # print(self.world['lidar'])
                if self.lidar_line is None:
                    self.lidar_line, = self.ax.plot(self.world['lidar'][:, 0], self.world['lidar'][:, 1],
                                                    marker=".",
                                                    markersize=2,
                                                    linewidth=0,
                                                    linestyle=None,
                                                    markerfacecolor='red',
                                                    markeredgecolor='red',
                                                    )
                else:
                    self.lidar_line.set_xdata(self.world['lidar'][:, 0])
                    self.lidar_line.set_ydata(self.world['lidar'][:, 1])
        except:
            pass

    def draw_crash(self):
        plt.text(0.1, 0.1, "CRASH!!!", size=50,
                 ha="center", va="center",
                 bbox=dict(facecolor = 'red')
                 )

    def add_quad(self, key, props):
        if not key in self.quads:
            self.quads[key] = props
        self.quads[key]['l1'], = self.ax.plot([], [], color='blue', linewidth=3, antialiased=False)
        self.quads[key]['l2'], = self.ax.plot([], [], color='red', linewidth=3, antialiased=False)
        self.quads[key]['hub'], = self.ax.plot([], [], marker='o', color=self.quads[key]['color'], markersize=self.quads[key]['markersize'], antialiased=False)

    def init_plot(self):
        for key in self.quads:
            self.quads[key]['l1'], = self.ax.plot([],[],color='blue',linewidth=3,antialiased=False)
            self.quads[key]['l2'], = self.ax.plot([],[],color='red',linewidth=3,antialiased=False)
            self.quads[key]['hub'], = self.ax.plot([],[],marker='o',color='green', markersize=6,antialiased=False)
        # Draw the path
        path = self.world["path"]
        self.line, = self.ax.plot(path[:,0], path[:,1], marker=".", linestyle="--", markersize=10, color='C7')

    def update(self):
        if self.world["map_size"] != self.map_size:
            self.map_size = self.world["map_size"]
            self.init_map_size()

        first_quad = None
        for key in self.quads:
            if first_quad is None:
                first_quad = key
            R = self.rotation_matrix(self.quads[key]['orientation'][0])
            L = self.quads[key]['L']
            points = np.array([ [-L,0], [L,0], [0,-L], [0,L], [0,0], [0,0] ]).T
            points = np.dot(R,points)
            points[0,:] += self.quads[key]['position'][0]
            points[1,:] += self.quads[key]['position'][1]
            self.quads[key]['l1'].set_data(points[0,0:2],points[1,0:2])
            self.quads[key]['l2'].set_data(points[0,2:4],points[1,2:4])
            self.quads[key]['hub'].set_data(points[0,5],points[1,5])
            # Draw the path if it hasnt updated
        path = self.world['path']
        if not np.array_equal(path, self.last_draw_path):
            self.draw_path(path)

        # Draw the obstacles
        current_height = self.quads[first_quad]['position'][2]
        if abs(current_height - self.last_draw_height) > 0.5:
            self.draw_world(current_height)



        if self.crashed:
            self.draw_crash()

        self.fig.canvas.draw()
        plt.pause(0.01)