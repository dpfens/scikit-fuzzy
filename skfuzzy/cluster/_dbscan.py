"""
Fuzzy DBSCAN algorithms for handling partial membership to clusters,
and overlapping clusters

"""
import math
from scipy.spatial import KDTree


class FuzzyCoreDBSCAN(object):
    """
    Fuzzy Core DBSCAN clustering algorithm [1].

    Relaxes the constraint of full membership imposed by DBSCAN by generating
    clusters with fuzzy core points.  Has applications when only an approximate
    density can be provided for defining clusters.

    References
    ----------
    .. [1] D. Ienco, G. Bordogna, Fuzzy extensions of the DBScan clustering
           algorithm,Soft Comput. (2016) 1–12.
    """

    def __init__(self, eps, min_points, max_points, tree_cls=KDTree):
        """
        Parameters
        ----------
        eps: float
            Maximum distance to be considered a neighbor of a point
        min_points : int
            minimum number of neighbors needed to have non-zero membership of a
            clusters
        max_points : int
            minimum number of neighbors needed to have full membership of a
            cluster
        tree_cls: class
            A class with a query method, that will return the nearest neighbors
            of a point within a given distance. Defaults to
            scipy.spatial.KDTree
        """
        if not isinstance(eps, (float, int)):
            raise TypeError('eps(%r) must be an integer or a float' % eps)

        if not isinstance(min_points, int):
            raise TypeError('min_points(%r) must be an integer' % min_points)
        if not isinstance(max_points, int):
            raise TypeError('max_points(%r) must be an integer' % max_points)

        if min_points > max_points:
            raise ValueError('min_points(%r) must be less than or equal to max_points(%r)' % (min_points, max_points))

        self.eps = eps
        self.min_points = min_points
        self.max_points = max_points
        self.tree_cls = tree_cls

    def fit_transform(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X : 2d array, size (S, N)
            Data to be clustered.  N is the number of data sets; S is the number
            of features within each sample vector.

        Returns
        -------
        tuple:
            list: A list of integers indicating the membership of the point
            list: A list of integers reflecting the type of membership, 0 is
                  an outlier, 1 is a core point
            list: A list of floats reflecting the fuzzy membership to a cluster
        """
        visited = set()
        cluster_id = 0

        count = len(X)
        indices = range(count)
        unvisited = set(indices)
        memberships = [1 for _ in range(count)]
        clusters = [None for _ in range(count)]
        point_types = [None for _ in range(count)]

        if isinstance(self.min_points, float):
            if self.min_points < 0 or self.min_points > 1:
                raise ValueError('min_points (%r) must be an integer or between 0 and 1' % self.min_points)
            min_points = math.round(self.min_points * count)
        else:
            min_points = self.min_points

        if isinstance(self.max_points, float):
            if self.max_points < 0 or self.max_points > 1:
                raise ValueError('max_points (%r) must be an integer or between 0 and 1' % self.max_points)
            max_points = math.round(self.max_points * count)
        else:
            max_points = self.max_points

        tree = self.tree_cls(X)

        while unvisited:
            point = unvisited.pop()
            visited.add(point)
            point_value = X[point]
            neighborhood = set(tree.query_ball_point(point_value, r=self.eps))
            if len(neighborhood) <= min_points:
                clusters[point] = -1
                point_types[point] = -1
                continue
            cluster_id += 1
            self.expand(point, X, tree, cluster_id, neighborhood, clusters, memberships, self.eps, min_points, max_points, point_types, visited)
            unvisited -= visited
        return clusters, point_types, memberships

    @staticmethod
    def membership(neighborhood, min_points, max_points):
        neighbor_count = len(neighborhood)
        if neighbor_count >= max_points:
            return 1
        elif neighbor_count <= min_points:
            return 0
        difference = max_points - min_points
        return float(neighbor_count - min_points) / difference

    def expand(self, index, X, tree, cluster_id, neighborhood, clusters, memberships, eps, min_points, max_points, point_types, visited):
        membership = self.membership(neighborhood, min_points, max_points)
        clusters[index] = cluster_id
        memberships[index] = membership
        point_types[index] = 0
        while neighborhood:
            neighbor = neighborhood.pop()
            if neighbor in visited:
                continue
            visited.add(neighbor)
            neighbor_point = X[neighbor]
            n_neighborhood = set(tree.query_ball_point(neighbor_point, r=eps))
            if len(n_neighborhood) > min_points:
                neighborhood |= n_neighborhood - visited
                n_membership = self.membership(n_neighborhood, min_points, max_points)
                clusters[neighbor] = cluster_id
                memberships[neighbor] = n_membership
                point_types[neighbor] = 0
            is_unassigned = clusters[neighbor] is None or clusters[neighbor] == -1
            if is_unassigned:
                min_membership = float('inf')
                for n_neighbor in n_neighborhood:
                    n_neighbor_point = X[n_neighbor]
                    n_neighbor_neighborhood = set(tree.query_ball_point(n_neighbor_point, r=eps))
                    n_neighbor_membership = self.membership(n_neighbor_neighborhood, min_points, max_points)
                    if n_neighbor_membership > 0 and n_neighbor_membership < min_membership:
                        min_membership = n_neighbor_membership
                memberships[neighbor] = min_membership
                point_types[neighbor] = 1
                clusters[neighbor] = cluster_id


class FuzzyBorderDBSCAN(object):
    """
    Fuzzy Border DBSCAN clustering algorithm [1].

    Relaxes the constraint of the epsilon parameter to allow clusters with
    overlapping borders, or, points that have partial membership to 1 or more
    clusters.  Has applications when only an approximate local neighborhood
    size can be provided, but crisp core memberships are still needed

    Notes
    -----
    The algorithm implemented is from [1]_.

    References
    ----------
    .. [1] D. Ienco, G. Bordogna, Fuzzy extensions of the DBScan clustering
           algorithm,Soft Comput. (2016) 1–12.
    """

    def __init__(self, min_points, min_eps, max_eps, tree_cls=KDTree):
        """
        Parameters
        ----------
        min_points : int
            minimum number of neighbors to be considered a cluster
        min_eps: float
            Maximum distance to have full membership of a cluster. Also is the
            maximum distance to be considered a neighbor of a point
        max_eps: float
            Maximum distance to have any membership of a cluster
        tree_cls: class
            A class with a query method, that will return the nearest neighbors
            of a point within a given distance. Defaults to
            scipy.spatial.KDTree
        """
        if not isinstance(min_points, int):
            raise TypeError('min_points(%r) must be an integer' % min_points)

        if not isinstance(min_eps, (float, int)):
            raise TypeError('min_eps(%r) must be an integer or a float' % min_eps)
        if not isinstance(max_eps, (float, int)):
            raise TypeError('max_eps(%r) must be an integer or a float' % max_eps)

        if min_eps > max_eps:
            raise ValueError('min_eps(%r) must be less than or equal to max_eps(%r)' % (min_eps, max_eps))

        self.min_points = min_points
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.tree_cls = tree_cls

    def fit_transform(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X : 2d array, size (S, N)
            Data to be clustered.  N is the number of data sets; S is the number
            of features within each sample vector.

        Returns
        -------
        tuple:
            list: A list of integers reflecting the type of membership, 0 is
                  an outlier, 1 is a core point
            list: A list of dict reflecting the fuzzy membership to clusters
        """
        visited = set()
        cluster_id = 0

        count = len(X)
        indices = range(count)
        unvisited = set(indices)
        memberships = [dict() for _ in range(count)]
        clusters = [None for _ in range(count)]
        point_types = [None for _ in range(count)]

        tree = self.tree_cls(X)

        while unvisited:
            point = unvisited.pop()
            visited.add(point)

            point_value = X[point]
            neighborhood = set(tree.query_ball_point(point_value, r=self.min_eps))
            if len(neighborhood) <= self.min_points:
                clusters[point] = -1
                point_types[point] = -1
                continue
            cluster_id += 1
            self.expand(point, X, tree, cluster_id, neighborhood, clusters, memberships, point_types, self.min_points, self.min_eps, self.max_eps, visited)
            unvisited -= visited
        return point_types, memberships

    @staticmethod
    def membership(distance, min_eps, max_eps):
        if distance <= min_eps:
            return 1
        elif distance > max_eps:
            return 0
        difference = max_eps - min_eps
        return float(max_eps - distance) / difference

    @classmethod
    def expand(cls, index, X, tree, cluster_id, neighborhood, clusters, memberships, point_types, min_points, min_eps, max_eps, visited):
        core_points = set()
        core_points.add(index)
        clusters[index] = cluster_id
        memberships[index] = {cluster_id: 1}
        point_types[index] = 0
        point = X[index]
        fuzzy_border_points = set(tree.query_ball_point(point, r=max_eps)) - neighborhood
        while neighborhood:
            neighbor = neighborhood.pop()
            if neighbor in visited:
                continue
            visited.add(neighbor)
            neighbor_point = X[neighbor]
            n_neighborhood = set(tree.query_ball_point(neighbor_point, r=min_eps))
            if len(n_neighborhood) > min_points:
                neighborhood |= n_neighborhood - visited
                n_fuzzy_border_points = set(tree.query_ball_point(neighbor_point, r=max_eps)) - n_neighborhood
                fuzzy_border_points |= n_fuzzy_border_points
                core_points.add(neighbor)
                memberships[neighbor] = {cluster_id: 1}
                point_types[neighbor] = 0
            else:
                fuzzy_border_points.add(neighbor)

        fuzzy_border_points -= core_points
        for point in fuzzy_border_points:
            point_value = X[point]
            distances, point_neighbors = tree.query(point_value, k=len(X), distance_upper_bound=max_eps)
            inf = float('inf')
            min_membership = inf
            for point_neighbor, point_distance in zip(point_neighbors, distances):
                distance_membership = cls.membership(point_distance, min_eps, max_eps)
                if point_neighbor in core_points and (point_distance < inf and distance_membership > 0):
                    min_membership = min(min_membership, distance_membership)

            if min_membership < inf:
                clusters[point] = cluster_id
                memberships[point][cluster_id] = min_membership
                point_types[point] = 1


class FuzzyDBSCAN(object):
    """
    Fuzzy DBSCAN clustering algorithm [1].

    Combines Fuzzy Core DBSCAN and Fuzzy Border DBSCAN to generate clusters
    with both fuzzy core memberships and fuzzy overlapping borders.

    Notes
    -----
    The algorithm implemented is from [1]_.

    References
    ----------
    .. [1] D. Ienco, G. Bordogna, Fuzzy extensions of the DBScan clustering
           algorithm,Soft Comput. (2016) 1–12.
    """

    def __init__(self, min_eps, max_eps, min_points, max_points, tree_cls=KDTree):
        """
        Parameters
        ----------
        min_eps: float
            Maximum distance to have full membership of a cluster.  Also is the
            maximum distance to be considered a neighbor of a point
        max_eps: float
            Maximum distance to have any membership of a cluster
        min_points : int
            minimum number of neighbors needed to have non-zero membership of a
            clusters
        max_points : int
            minimum number of neighbors needed to have full membership of a
            cluster
        tree_cls: class
            A class with a query method, that will return the nearest neighbors
            of a point within a given distance. Defaults to
            scipy.spatial.KDTree
        """
        if not isinstance(min_eps, (float, int)):
            raise TypeError('min_eps(%r) must be an integer or a float' % min_eps)
        if not isinstance(max_eps, (float, int)):
            raise TypeError('max_eps(%r) must be an integer or a float' % max_eps)

        if min_eps > max_eps:
            raise ValueError('min_eps(%r) must be less than or equal to max_eps(%r)' % (min_eps, max_eps))

        if not isinstance(min_points, int):
            raise TypeError('min_points(%r) must be an integer' % min_points)
        if not isinstance(max_points, int):
            raise TypeError('max_points(%r) must be an integer' % max_points)

        if min_points > max_points:
            raise ValueError('min_points(%r) must be less than or equal to max_points(%r)' % (min_points, max_points))

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.min_points = min_points
        self.max_points = max_points
        self.tree_cls = tree_cls

    def fit_transform(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X : 2d array, size (S, N)
            Data to be clustered.  N is the number of data sets; S is the number
            of features within each sample vector.

        Returns
        -------
        tuple:
            list: A list of integers reflecting the type of membership, 0 is
                  an outlier, 1 is a core point
            list: A list of dict reflecting the fuzzy membership to clusters
        """
        visited = set()
        cluster_id = 0

        count = len(X)
        indices = range(count)
        unvisited = set(indices)
        memberships = [dict() for _ in range(count)]
        point_types = [None for _ in range(count)]

        tree = self.tree_cls(X)

        while unvisited:
            point = unvisited.pop()
            visited.add(point)
            point_value = X[point]
            raw_distances, raw_neighborhood = tree.query(point_value, k=len(X), distance_upper_bound=self.max_eps)
            distances = []
            neighborhood = []
            for i in range(len(raw_neighborhood)):
                if raw_distances[i] < float('inf'):
                    distances.append(raw_distances[i])
                    neighborhood.append(raw_neighborhood[i])
            density = self.density(distances, self.min_eps, self.max_eps)
            point_membership = self.core_membership(density, self.min_points, self.max_points)
            if not point_membership:
                point_types[point] = -1
                continue
            cluster_id += 1
            memberships[point][cluster_id] = point_membership
            point_types[point] = 0
            neighborhood = set(neighborhood)
            self.expand(X, tree, point, cluster_id, neighborhood, memberships, point_types, self.min_points, self.max_points, self.min_eps, self.max_eps, visited)
            unvisited -= visited
        return point_types, memberships

    @staticmethod
    def distance_membership(distance, min_eps, max_eps):
        if distance <= min_eps:
            return 1
        elif distance > max_eps:
            return 0
        difference = max_eps - min_eps
        return float(max_eps - distance) / difference

    @staticmethod
    def core_membership(n, min_points, max_points):
        if n >= max_points:
            return 1
        elif n <= min_points:
            return 0
        difference = max_points - min_points
        return float(n - min_points) / difference

    @classmethod
    def density(cls, distances, min_eps, max_eps):
        return sum(cls.distance_membership(distance, min_eps, max_eps) for distance in distances)

    @classmethod
    def expand(cls, X, tree, index, cluster_id, neighborhood, memberships, point_types, min_points, max_points, min_eps, max_eps, visited):
        core_points = set([index])
        border_points = set()
        while neighborhood:
            neighbor = neighborhood.pop()
            visited.add(neighbor)
            neighbor_value = X[neighbor]
            raw_distances, raw_n_neighborhood = tree.query(neighbor_value, k=len(X), distance_upper_bound=max_eps)

            distances = []
            n_neighborhood = []
            for i in range(len(raw_n_neighborhood)):
                if raw_distances[i] < float('inf'):
                    distances.append(raw_distances[i])
                    n_neighborhood.append(raw_n_neighborhood[i])
            n_density = cls.density(distances, min_eps, max_eps)
            n_membership = cls.core_membership(n_density, min_points, max_points)
            if n_membership > 0:
                core_points.add(neighbor)
                neighborhood |= set(n_neighborhood)
                memberships[neighbor][cluster_id] = n_membership
                point_types[neighbor] = 0
            else:
                border_points.add(neighbor)
            neighborhood -= core_points

        for border_point_index in border_points:
            border_point = X[border_point_index]
            raw_distances, raw_border_neighborhood = tree.query(border_point, k=len(X), distance_upper_bound=max_eps)
            min_membership = 1.0
            for i in range(len(raw_border_neighborhood)):
                if raw_distances[i] < max_eps and raw_border_neighborhood[i] in core_points:
                    distance_membership = cls.distance_membership(raw_distances[i], min_eps, max_eps)
                    core_neighbor_point = raw_border_neighborhood[i]
                    min_membership = min(distance_membership, memberships[core_neighbor_point][cluster_id], min_membership)
            memberships[border_point_index][cluster_id] = min_membership
            point_types[neighbor] = 1
