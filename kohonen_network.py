import numpy as np
from shapely.geometry import Point, Polygon
from plotly.offline import plot
from fastdist import fastdist
import plotly.graph_objects as go


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
        closest_index, closest_distance = 0, fastdist.euclidean(np.array(self.get_coords()),
                                                                np.array(neurons[0].get_coords))
        for index, point in enumerate(neurons):
            if dist := fastdist.euclidean(np.array(self.get_coords()), np.array(point.get_coords)) < closest_distance:
                closest_index, closest_distance = index, dist
        return closest_index


class SOM:
    """
    SOM model
    :param radius: radius to find neurons'neighbors
    :param number_of_points: number of data points to generate
    :param shape: shape of the data, one of ['square', 'triangle', 'random']

    :var polygon: Polygon Object, use to create data points
    :var data_points: list of MyPointClass Object
    :var neuron_points: numpy array of size 5x5 with MyPointClass Object
    """

    def __init__(self, radius: float, number_of_points: int, shape: str = 'square'):
        self.number_of_points = number_of_points
        self.shape = shape
        self.radius = np.sqrt(number_of_points)/2
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
        step_x, step_y = (maxx - minx) * 0.025, (maxy - miny) * 0.025
        neuron_points = np.array([[MyPointClass(np.array(mean_x - 2 * step_x + i * step_x, dtype=object),
                                                np.array(mean_y + 2 * step_y - j * step_y, dtype=object))
                                   for i in range(5)] for j in range(5)])
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
        xneuron, yneuron = [point.x for point in self.neuron_points.flatten()], \
                           [point.y for point in self.neuron_points.flatten()]
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
            fig.add_scatter(x=xpolygon, y=ypolygon, mode='lines',
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
                                    opacity=0.5),
                        name="polygon")
        # Plot the list of points
        fig.add_scatter(x=xdata, y=ydata, name="points", mode='markers',
                        marker=dict(color='blue',
                                    line_width=1.5,
                                    size=10,
                                    opacity=0.5))

        plot(fig)

    def get_index_closest_neigbours(self, index):
        """
        :param index: index of the neuron
        :return: indexes of neurons' closest at max radius distance
        """
        ...

    def move_closest_neuron_and_neighbours(self, index):
        """
        move all neurons concerned
        :param index: index of the actual closest neuron
        """
        ...

    def fit(self):
        for input_data in self.data_points:
            # On parcourt les données d'entrées
            # On cherche le neurone le plus proche
            # On rapproche ce neurone le plus proche de l'input data ainsi que les voisins de ce dernier
            ...
