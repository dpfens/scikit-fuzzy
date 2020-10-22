import unittest
import skfuzzy as fuzz


class TestFuzzyDBSCAN(unittest.TestCase):

    def setUp(self):
        self.points = [
            (-20.9212126798201, 44.6617417719858),
            (-20.724328120394947, 39.95452070941839),
            (-8.66481085331423, 49.40107442975715),
            (-7.344393815661689, 50.79249671704422),
            (-17.006554554596583, 33.707979409343984),
            (-17.101555311425013, 56.67222003895794),
            (-3.1905121947270914, 46.53590908122198),
            (-8.796842136512238, 45.252489295057856),
            (-3.2376662399886094, 39.53817779023757),
            (-6.3655584837861765, 49.11399755780584),
            (-22.761614893194764, 50.51507191229298),
            (-10.260792264195635, 44.960442921023116),
            (-5.916717851365984, 55.084261276399246),
            (-8.68921078126392, 38.64124297247835),
            (-6.465124264878238, 35.530231140018024),
            (-11.929837686839782, 31.950568664939112),
            (-12.190650274244112, 32.00296891760979),
            (-1.8627399190885896, 55.49188559633143),
            (-15.946374684065471, 50.970940281396764),
            (-3.3036981333507427, 33.106322911678575),
            (-8.000623894629777, 44.203129957183734),
            (-10.159283736317224, 37.30211558886747),
            (-10.685736565293778, 40.51188437889208),
            (-5.052015748633659, 48.22762177005775),
            (-11.201286497455538, 30.944546772841946),
            (-25.859567633887842, 46.14378904207943),
            (-3.301511811028538, 42.553478899948516),
            (-14.369906097827581, 37.281880699552595),
            (-14.263785442046148, 44.214757501554665),
            (1.8101007896395327, 44.69473288092371),
            (-23.26495687884722, 28.94832028302858),
            (-13.351131196550899, 44.33995896626044),
            (-9.175521929808312, 45.3129694776244),
            (-16.996228321595027, 35.71259137992261),
            (-12.935704419930008, 43.406598890376756),
            (-24.01582629061314, 42.79950320038263),
            (-2.255544587810574, 44.30126220139316),
            (-3.7173618961814903, 31.1168006806653),
            (-13.74148194234462, 44.97490203148087),
            (-4.81520978218526, 35.40260679308956),
            (-16.915153906415327, 45.74519576319235),
            (-10.605369328223702, 41.819021648071235),
            (-26.24401470022997, 34.61898671145311),
            (-11.530688640488027, 46.746619853944296),
            (-19.17004451047878, 39.62840373226089),
            (-16.565145633607184, 45.069669947287814),
            (1.2883653154924435, 46.59978355553689),
            (-14.022029438709598, 45.98403539198285),
            (-4.812355142680133, 68.56754625237589),
            (-8.197901301586652, 61.17595303833182),
            (12.244846318164893, 14.709391211010898),
            (14.452561685593215, 8.588175655494734),
            (14.849174313552622, 1.4518519851403582),
            (9.954654653756684, -2.090598112256515),
            (3.0614245954681305, 17.420578374018646),
            (11.449020323035711, -1.846895591981605),
            (18.674231911088594, -1.6520721981804591),
            (10.751954181213332, -5.276583311277536),
            (17.614003862641166, 4.485650478031118),
            (13.021680622512308, -2.305801133707604),
            (20.4948183410774, 6.227971432098055),
            (23.792588128438574, 8.092720251417504),
            (24.021809429825872, 6.860688636928994),
            (10.918812348384709, 0.2024292887435548),
            (9.3142858251294, 11.519263586103035),
            (8.115046499207132, -2.188444858647447),
            (20.21761049550352, 1.432510740352341),
            (2.9216815225497665, 19.227704660669698),
            (19.503369479185952, 1.4447521719095509),
            (14.433358551327114, 6.620196936805936),
            (-2.0367697328197885, 23.130803576101727),
            (13.850071350173003, 9.714767104986654),
            (9.815630085209296, 11.482180998420354),
            (21.252867648955462, 16.988238540124875),
            (11.741608217898685, 14.970261970779651),
            (19.74304949845673, 9.164188205059796),
            (15.971724195013557, 17.708943659159868),
            (8.701073594356608, 4.048840513151777),
            (11.161450802976997, 11.837946869577312),
            (12.461450184125942, -3.450613468838311),
            (0.6031713720916585, 1.0484011296645823),
            (17.86375222872701, 16.572542438970686),
            (17.96180523431209, 0.23046452291564457),
            (5.351609722922982, 16.124790977257987),
            (10.77392935885976, 4.420851750059285),
            (12.636542809657259, -4.898157203839073),
            (18.71154760327175, 9.08002662560643),
            (-0.485222835717547, 7.192563794453096),
            (2.390338476674197, 25.4748505316751),
            (-3.446676451817172, -8.026334303401562),
            (15.191087828055007, 16.114290506289947),
            (25.66571234143907, 13.0661956006332),
            (11.414022785680881, 12.623265348078707),
            (20.70716366242688, 0.898276535218776),
            (8.362831897931196, 6.024286630048872),
            (9.938118538011267, -2.3939539692105605),
            (16.38076923044898, -0.6395654904701704),
            (12.598179074434421, -10.268889154481592),
            (18.226634394761707, -0.9152048270401636),
            (16.63628454485014, 0.6347701557103633)
        ]

    def test_fuzzycore(self):
        """
        Test fuzzy core DBSCAN
        """
        eps = 8
        min_points = 7
        max_points = 15

        clf = fuzz.cluster.FuzzyCoreDBSCAN(eps, min_points, max_points)
        clusters, types, memberships = clf.fit_transform(self.points)

        cluster_counts = dict()
        for cluster, membership in zip(clusters, memberships):
            cluster_counts.setdefault(cluster, 0)
            cluster_counts[cluster] += 1
            self.assertTrue(0 <= membership <= 1)

        expected_counts = {
            1: 48,
            2: 45,
            -1: 7
        }
        for key in expected_counts:
            self.assertEqual(cluster_counts[key], expected_counts[key])

    def test_fuzzyborder(self):
        """
        Test fuzzy border DBSCAN
        """
        min_points = 5
        min_eps = 6
        max_eps = 25
        clf = fuzz.cluster.FuzzyBorderDBSCAN(min_points, min_eps, max_eps)
        types, memberships = clf.fit_transform(self.points)
        cluster_counts = dict()
        for membership in memberships:
            clusters = list(membership.keys())
            clusters.sort()
            clusters = tuple(clusters)
            cluster_counts.setdefault(clusters, 0)
            cluster_counts[clusters] += 1

            for cluster in clusters:
                self.assertTrue(0 <= membership[cluster] <= 1)

        expected_counts = {
            (1, ): 48,
            (2, ): 45,
            (1, 2): 7
        }
        for key in expected_counts:
            self.assertEqual(cluster_counts[key], expected_counts[key])

    def test_fuzzydbscan(self):
        """
        Test fuzzy DBSCAN
        """
        min_points = 8
        max_points = 20
        min_eps = 5
        max_eps = 14

        clf = fuzz.cluster.FuzzyDBSCAN(min_eps, max_eps, min_points, max_points)
        types, memberships = clf.fit_transform(self.points)

        cluster_counts = dict()
        for membership in memberships:
            clusters = list(membership.keys())
            clusters.sort()
            clusters = tuple(clusters)
            cluster_counts.setdefault(clusters, 0)
            cluster_counts[clusters] += 1

            for cluster in clusters:
                self.assertTrue(0 <= membership[cluster] <= 1)

        expected_counts = {
            (1, ): 50,
            (2, ): 47,
            (1, 2): 3
        }
        for key in expected_counts:
            self.assertEqual(cluster_counts[key], expected_counts[key])
