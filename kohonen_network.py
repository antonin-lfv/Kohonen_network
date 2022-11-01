import numpy as np
from shapely.geometry import Point, Polygon
from plotly.offline import plot
from fastdist import fastdist
import plotly.graph_objects as go
import random
import time


class MyPointClass:
    """
    Custom Point class
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coords(self):
        return [self.x, self.y]

    def get_closest(self, neurons):
        """
        Find the closest neuron
        :param neurons : neurons' list
        :return: closest neuron index in neurons' list
        """
        closest_index, closest_distance = 0, fastdist.euclidean(np.array(self.get_coords(), dtype=float),
                                                                np.array(neurons[0].get_coords(), dtype=float))
        for index, point in enumerate(neurons):
            if dist := fastdist.euclidean(np.array(self.get_coords(), dtype=float),
                                          np.array(point.get_coords(), dtype=float)) < closest_distance:
                closest_index, closest_distance = index, dist
        return closest_index

    def move_to(self, neuron, scale):
        """
        Move the point to the neuron
        :param scale: scale of the movement
        :param neuron: neuron to move to
        """
        self.x += (neuron.x - self.x) * scale
        self.y += (neuron.y - self.y) * scale


class SOM:
    """
    SOM model
    :param number_of_points: number of data points to generate
    :param shape: shape of the data, one of ['square', 'triangle', 'random']

    :var polygon: Polygon Object, use to create data points
    :var data_points: list of MyPointClass Object
    :var neuron_points: 1-D numpy array with MyPointClass Object
    :var radius: radius to find neurons'neighbors
    :var pourcent_distance_closest: how much the closest neuron move to the input
    :var pourcent_distance_closest_neighbors: how much the neighbors of the closest neuron move to the input
    """

    def __init__(self, number_of_points: int, shape: str = 'square'):
        self.number_of_points = number_of_points
        self.shape = shape
        self.radius = 0.2
        self.pourcent_distance_closest = 0.8
        self.pourcent_distance_closest_neighbors = 0.25
        self.polygon, self.data_points, self.neuron_points = self.__get_polygon_from_shape()

    def __get_polygon_from_shape(self):
        """
        Create the data from the given shape
        :return: polygon object and list of Point objects
        """
        assert isinstance(self.shape, str), print("shape must be one of ['square', 'triangle', 'random']")
        polygon, neuron_points = None, None
        if self.shape == 'square':
            polygon = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        elif self.shape == 'triangle':
            polygon = Polygon([[0, 0], [0.5, 4], [1, 0]])
        elif self.shape == 'random':
            polygon = Polygon([[0, 0], [0, 2], [1.5, 1], [0.5, -0.5], [0, 0]])
        data_points = self.__Random_Points_in_Polygon(polygon)
        minx, miny, maxx, maxy = polygon.bounds
        mean_x, mean_y = (minx + maxx) / 2, (miny + maxy) / 2
        step_x, step_y = (maxx - minx) * 0.04, (maxy - miny) * 0.04
        neuron_points = np.array([[MyPointClass(np.array(mean_x - 2 * step_x + i * step_x, dtype=float),
                                                np.array(mean_y + 2 * step_y - j * step_y, dtype=float))
                                   for i in range(5)] for j in range(5)]).flatten()
        return polygon, data_points, neuron_points

    def __Random_Points_in_Polygon(self, polygon):
        points = []
        minx, miny, maxx, maxy = polygon.bounds
        while len(points) < self.number_of_points:
            x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
            # we use the Point class of shapely.geometry to use the contains method and create our data
            pnt = Point(x, y)
            if polygon.contains(pnt):
                points.append(MyPointClass(x, y))
        return points

    def get_all_data(self):
        """
        :return: xpolygon, ypolygon,xdata, ydata as python lists
        """
        xpolygon, ypolygon = self.polygon.exterior.xy
        xdata, ydata = [point.x for point in self.data_points], [point.y for point in self.data_points]
        xneuron, yneuron = [point.x for point in self.neuron_points], \
                           [point.y for point in self.neuron_points]
        return xpolygon.tolist(), ypolygon.tolist(), xdata, ydata, xneuron, yneuron

    def get_neurons(self):
        """
        :return: 5x5 numpy array of neurons' points
        """
        return self.neuron_points

    def display_data(self, display_shape=False):
        xpolygon, ypolygon, xdata, ydata, xneuron, yneuron = self.get_all_data()
        fig = go.Figure()
        # Plot the polygon
        if display_shape:
            fig.add_scatter(x=xpolygon,
                            y=ypolygon,
                            mode='lines',
                            marker=dict(color='green',
                                        line_width=0.1,
                                        size=1,
                                        opacity=0.1),
                            name="polygon")
        # Plot the neurons
        fig.add_scatter(x=xneuron,
                        y=yneuron,
                        mode='markers',
                        marker=dict(color='red',
                                    size=10,
                                    opacity=1),
                        text=[i for i in range(len(xneuron))],
                        texttemplate="%{text}",
                        name="Neurons")
        # TODO Plot the line between neurons
        # Plot the list of points
        fig.add_scatter(x=xdata, y=ydata,
                        name="Data",
                        mode='markers',
                        marker=dict(color='blue',
                                    line_width=1.5,
                                    size=10,
                                    opacity=0.4))

        plot(fig)

    def get_index_closest_neigbours(self, index):
        """
        :param index: index of the neuron
        :return: indexes of neurons' closest at max radius distance
        """
        closest_neigbours = []
        for index_neuron, neuron in enumerate(self.neuron_points):
            if index_neuron != index:
                if fastdist.euclidean(np.array(self.neuron_points[index].get_coords(), dtype=float),
                                      np.array(neuron.get_coords(), dtype=float)) < self.radius:
                    closest_neigbours.append(index_neuron)
        return closest_neigbours

    def move_closest_neuron_and_neighbours(self, index_closest_neuron, index_input_data, t):
        """
        move all neurons concerned
        :param index_input_data: index of the input data
        :param index_closest_neuron: index of the actual closest neuron
        :param t: iteration
        """
        # first we find the neighbors of the neuron
        indexes = self.get_index_closest_neigbours(index_closest_neuron)
        # We move the closest neuron to the input data with pourcent_distance_closest of their distance
        self.neuron_points[index_closest_neuron].move_to(self.data_points[index_input_data],
                                                         self.pourcent_distance_closest)
        # We move the closest neurons of the closest neuron to the input data with
        # pourcent_distance_closest_neighbors of their distance
        for indexes_neigh_of_neigh in indexes:
            self.neuron_points[indexes_neigh_of_neigh].move_to(self.data_points[index_input_data], self.pourcent_distance_closest_neighbors)

    def fit(self, epochs: int = 100):
        """
        Run the model
        """
        for t in range(epochs):
            # We go through the input data
            random.shuffle(self.data_points)
            for index_input_data, input_data in enumerate(self.data_points):
                # We find the closest neuron of the input data (index)
                index_closest_neuron_of_input = input_data.get_closest(neurons=self.neuron_points)
                # We move the closest neuron and its neighbours
                self.move_closest_neuron_and_neighbours(index_closest_neuron_of_input, index_input_data, t)
                self.display_data()
                time.sleep(2)
            # We decrease the radius
            self.radius *= 0.95


if __name__ == "__main__":
    som_model = SOM(number_of_points=1000, shape='square')
    som_model.fit()
    som_model.display_data()
