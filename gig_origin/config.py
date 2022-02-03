class MLConfig:
    def __init__(self, arg):
        self.arg = arg
        dataset = self.arg.dataset
        # Possible versions: 'GraphConv', 'GIN'
        self.node_level_module = self.arg.node_level_module
        # Main features to change!!!!
        # if True uses DGCNN, if False uses FixedGraph, LGL, LGLKL
        self.DGCNN = self.arg.DGCNN
        # If '' and DGCNN=False and model='', then LGL.
        # If 'KL_degree_dist' and DGCNN=False and model='', then LGLKL.
        self.regularization = self.arg.regularization
        # 'FixedGraphInGraph' - uses Fixed graphs, precomputed
        # 'GCN' - just normal GCN model
        # '' - uses GraphInGraph model
        self.model = self.arg.model
        # Version of fixed population graphs: 'knn', 'random'
        self.fix_population_graph = self.arg.fix_population_graph

        print(dataset)
        if dataset == "PROTEINS_full":
            self.dataset = dataset
            self.batch_size_train = 50
            self.batch_size_val = 64
            self.batch_size_test = 50
            self.lr = 1e-3# wo kl

            self.sigma_lr =1e-2 # should be the same as lr
            self.mu_lr =1e-2
            self.theta_lr = 1e-2
            self.temp_lr = 1e-2
            self.temp_initial =1e-4
            self.theta_initial = 1.0

            self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
            self.random_split = False

            # percentage of training set to use as validation
            self.n_epoch = 3000
            self.input_dim = (28 * 28)
            self.output_dim = 10
            self.random_seed = 0
            self.num_node_features = 1
            self.device = 'cuda'#'cpu'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50
            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'
            if self.regularization == 'KL_degree_dist':
                self.theta_initial = 8.0
            self.target_distribution = 'normal' #self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.003
                self.kl_sum_par_after_tresh = 0.003# 0.05 most working version 0.88 nwas with 0.02

            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3

            else:
                self.kl_sum_par = ''
            self.ce_sum_par = 1
            self.ce_sum_par_after_tresh = 1
            self.learnable_distr = True
            self.kl_treshold = 2
            self.fix_graph_random_num = 0 # should start from 0!
            self.sigma_reg = False
        elif dataset == 'Tox21':
            self.dataset = dataset
            self.n_epoch = 3000
            self.input_dim = None
            self.output_dim = None
            self.random_seed = 0
            self.num_node_features = None
            self.device = 'cuda'
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.patience = 50
            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'
            self.fix_graph_random_num = 0
            self.random_split = False
            self.label_idx = None# max11
            self.label_treshold = 0

            if self.label_idx == None:
                self.batch_size_train = 25 # 25
                self.batch_size_val = 25
                self.batch_size_test = 25
                if self.fix_population_graph == 'knn':
                    self.batch_size_train = 50  # 25
                    self.batch_size_val = 50
                    self.batch_size_test = 50
                self.lr = 1e-3
                self.sigma_lr = 1e-3  # gender1e-1# should be the same as lr
                self.mu_lr = 1e-3  # gender 1e-1
                self.theta_lr = 1e-3
                self.temp_lr = 1e-3
                self.patience = 50
                self.temp_initial = 1e-1
                self.theta_initial = 8.0
                self.pooling = 'add'#, add 'mean', 'attention', 'set2set'

                self.dropout = 0.15
                self.use_dropout = False
                self.target_distribution = 'normal'  # self.arg.target_distribution#'normal'# 'normal', 'power_law'
                if self.target_distribution == 'normal':
                    self.kl_sum_par = 0.0001# was for cosine learning 3
                    self.kl_sum_par_after_tresh = 0.0001#0.003  # 0.05 most working version 0.88 nwas with 0.02

                elif self.target_distribution == 'power_law':
                    self.kl_sum_par =  0.0001
                    self.kl_sum_par_after_tresh = 0.0001

                else:
                    self.kl_sum_par = ''
                self.kl_treshold = 3
                self.ce_sum_par = 1
                self.ce_sum_par_after_tresh = 1
                self.learnable_distr = True
                self.sigma_reg = False


        elif dataset == "HCP":
            self.dataset = dataset
            self.hcp_dim = 200
            # percentage of training set to use as validation
            self.n_epoch = 600
            self.input_dim = 200
            self.output_dim = 2
            self.random_seed = 0
            self.num_node_features = 200
            self.device = 'cuda'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = True
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.patience = 50
            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'
            self.task = 'gender'
            self.target_distribution = 'normal' #self.arg.target_distribution#'normal'# 'normal', 'power_law'
            self.random_split = False

            if self.task == 'gender':
                self.patience = 50
                self.batch_size_train = 25 # 10 was for gender, 14 for age
                self.batch_size_val = 64
                self.batch_size_test = 25
                if self.model == 'FixedGraphInGraph':
                    self.lr = 1e-2  # was 4#1e-3 for gender
                    self.n_epoch = 150

                else:
                    self.lr = 1e-3  # was 4#1e-3 for gender

                self.sigma_lr = 1e-1  # gender1e-1# should be the same as lr
                self.mu_lr = 1e-1  # gender 1e-1
                self.theta_lr = 1e-3
                self.temp_lr = 1e-3

                self.temp_initial = 1e-1
                self.theta_initial = 1.0
                self.pooling = 'mean'

                if self.target_distribution == 'normal':
                    self.kl_sum_par = 1 #0.009
                    self.kl_sum_par_after_tresh = 1#0.009
                elif self.target_distribution == 'power_law':
                    self.kl_sum_par = 0.3
                else:
                    self.kl_sum_par = ''

                self.kl_treshold = 5.0  # smaller
                self.ce_sum_par = 1
                self.ce_sum_par_after_tresh = 1

                self.learnable_distr = True



            self.fix_graph_random_num = 6
            self.sigma_reg = False
        elif dataset == "PPI":
            self.dataset = dataset
            self.batch_size_train = 30 # 64
            self.batch_size_val = 64
            self.batch_size_test = 30
            self.lr = 1e-5
            self.sigma_lr =1e-3# should be the same as lr
            self.mu_lr =1e-3
            self.theta_lr = 1e-3
            self.temp_lr = 1e-3

            self.temp_initial =1e-1
            self.theta_initial = 8.0

            # percentage of training set to use as validation
            self.n_epoch = 500
            self.input_dim = 1
            self.output_dim = 2
            self.random_seed = 0
            self.num_node_features = 1
            self.device = 'cuda'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50


            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'
            self.target_distribution = 'normal' #self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.09 # 0.0015
                self.kl_sum_par_after_tresh = 0.03 # 0.0015
            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3
            else:
                self.kl_sum_par = ''
            self.learnable_distr = True
            self.kl_treshold = 3
            self.fix_graph_random_num = 6
            self.sigma_reg = False

        elif dataset in ["PROTEINS"]:
            self.dataset = dataset
            self.batch_size_train = 50
            self.batch_size_val = 64
            self.batch_size_test = 50
            self.lr = 1e-3# wo kl

            self.sigma_lr =1e-2 # should be the same as lr
            self.mu_lr =1e-2
            self.theta_lr = 1e-2
            self.temp_lr = 1e-2
            self.temp_initial =1e-4
            self.theta_initial = 8.0#8.0

            self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
            self.random_split = True

            # percentage of training set to use as validation
            self.n_epoch = 3000
            self.input_dim = 3
            self.output_dim = 2
            self.random_seed = 0
            self.num_node_features = 3
            self.device = 'cuda'#'cpu'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50


            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'



            if self.regularization == 'KL_degree_dist':
                self.theta_initial = 1.0  # 8.0
            self.target_distribution = 'normal' #self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.003
                self.kl_sum_par_after_tresh = 0.003# 0.05 most working version 0.88 nwas with 0.02

            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3

            else:
                self.kl_sum_par = ''
            self.ce_sum_par = 1
            self.ce_sum_par_after_tresh = 1
            self.learnable_distr = True
            self.kl_treshold = 2
            self.fix_graph_random_num = 0 # should start from 0!
            self.sigma_reg = False
        elif dataset in ["DD"]:

            self.dataset = dataset
            self.batch_size_train = 50
            self.batch_size_val = 64
            self.batch_size_test = 50
            self.lr = 1e-3  # wo kl

            self.sigma_lr = 1e-2  # should be the same as lr
            self.mu_lr = 1e-2
            self.theta_lr = 1e-2
            self.temp_lr = 1e-2
            self.temp_initial = 1e-4
            self.theta_initial = 8.0  # 8.0

            self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
            self.random_split = True

            # percentage of training set to use as validation
            self.n_epoch = 3000
            self.input_dim = 89
            self.output_dim = 2
            self.random_seed = 0
            self.num_node_features = 89
            self.device = 'cuda'  # 'cpu'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50

            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'

            if self.regularization == 'KL_degree_dist':
                self.theta_initial = 1.0  # 8.0
            self.target_distribution = 'normal'  # self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.003
                self.kl_sum_par_after_tresh = 0.003  # 0.05 most working version 0.88 nwas with 0.02

            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3

            else:
                self.kl_sum_par = ''
            self.ce_sum_par = 1
            self.ce_sum_par_after_tresh = 1
            self.learnable_distr = True
            self.kl_treshold = 2
            self.fix_graph_random_num = 0  # should start from 0!
            self.sigma_reg = False
        elif dataset in ["NCI1"]:

            self.dataset = dataset
            self.batch_size_train = 50
            self.batch_size_val = 64
            self.batch_size_test = 50
            self.lr = 1e-3  # wo kl

            self.sigma_lr = 1e-2  # should be the same as lr
            self.mu_lr = 1e-2
            self.theta_lr = 1e-2
            self.temp_lr = 1e-2
            self.temp_initial = 1e-4
            self.theta_initial = 8.0  # 8.0

            self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
            self.random_split = True

            # percentage of training set to use as validation
            self.n_epoch = 3000
            self.input_dim = 89
            self.output_dim = 2
            self.random_seed = 0
            self.num_node_features = 37
            self.device = 'cuda'  # 'cpu'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50

            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'

            if self.regularization == 'KL_degree_dist':
                self.theta_initial = 1.0  # 8.0
            self.target_distribution = 'normal'  # self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.003
                self.kl_sum_par_after_tresh = 0.003  # 0.05 most working version 0.88 nwas with 0.02

            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3

            else:
                self.kl_sum_par = ''
            self.ce_sum_par = 1
            self.ce_sum_par_after_tresh = 1
            self.learnable_distr = True
            self.kl_treshold = 2
            self.fix_graph_random_num = 0  # should start from 0!
            self.sigma_reg = False
        elif dataset in ["ENZYMES"]:

            self.dataset = dataset
            self.batch_size_train = 50
            self.batch_size_val = 64
            self.batch_size_test = 50
            self.lr = 1e-3  # wo kl

            self.sigma_lr = 1e-2  # should be the same as lr
            self.mu_lr = 1e-2
            self.theta_lr = 1e-2
            self.temp_lr = 1e-2
            self.temp_initial = 1e-4
            self.theta_initial = 8.0  # 8.0

            self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
            self.random_split = True

            # percentage of training set to use as validation
            self.n_epoch = 3000
            self.input_dim = 21
            self.output_dim = 6
            self.random_seed = 0
            self.num_node_features = 21
            self.device = 'cuda'  # 'cpu'
            self.dropout = 0.15  # 0.15 # try 0 drop out
            self.use_dropout = False
            self.fix_population_level = False
            self.fix_node_level = True
            self.fix_dynamic_node_convolution = False
            self.writer_path = "runs/"
            self.kl_patience = 12
            self.ce_patience = 5
            self.patience = 50

            self.feature_dataset = False
            self.data_level = 'graph'  # 'image', 'feature_matrix','graph'

            if self.regularization == 'KL_degree_dist':
                self.theta_initial = 1.0  # 8.0
            self.target_distribution = 'normal'  # self.arg.target_distribution#'normal'# 'normal', 'power_law'
            if self.target_distribution == 'normal':
                self.kl_sum_par = 0.003
                self.kl_sum_par_after_tresh = 0.003  # 0.05 most working version 0.88 nwas with 0.02

            elif self.target_distribution == 'power_law':
                self.kl_sum_par = 0.3

            else:
                self.kl_sum_par = ''
            self.ce_sum_par = 1
            self.ce_sum_par_after_tresh = 1
            self.learnable_distr = True
            self.kl_treshold = 2
            self.fix_graph_random_num = 0  # should start from 0!
            self.sigma_reg = False
        else:
            raise NotImplementedError


