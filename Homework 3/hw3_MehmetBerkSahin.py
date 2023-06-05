
from ComputationalGraphPrimer import *
import random
import numpy as np
import operator
import matplotlib.pyplot as plt


class MyComputationalGraphPrimer(ComputationalGraphPrimer):
    ####################################################################################################################
    # I implemented this function (MBS)
    def __init__(self, optimizer='sgd', momentum=None, beta1=None, beta2=None, *args, **kwargs):
        super(MyComputationalGraphPrimer, self).__init__(*args, **kwargs)
        # set the parameters that are entered
        if optimizer:
            self.optimizer = optimizer
        if momentum:
            self.momentum = momentum
        if beta1:
            self.beta1 = beta1
        if beta2:
            self.beta2 = beta2
        # set necessary parameters depending on the optimizer
        if self.optimizer == "sgd+":
            # Keeps the updates for momentum SGD
            self.step_sizes = None
            self.bias_m = None
        elif self.optimizer == "adam":
            # bias corrected moments
            self.moment1 = None # corresponds to m
            self.moment2 = None # corresponds to v
            self.bias_m1 = None
            self.bias_m2 = None

    ####################################################################################################################

    # training loop for one neuron classifier
    def run_training_loop_one_neuron_model(self, training_data):
        """
        DISCLAIMER: I copied this function from Prof. Kak's ComputationalGraphPrimer source code.
        I marked the adjustments I made by adding MBS (Mehmet Berk Sahin) at the end of the comments.
        And I pointed the modifications out by showing them in a box with '#' symbol.

        The training loop must first initialize the learnable parameters.  Remember, these are the
        symbolic names in your input expressions for the neural layer that do not begin with the
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0, 1) for param in self.learnable_params}

        self.bias = random.uniform(0, 1)  ## Adding the bias improves class discrimination.

        ###################################################################
        # initialize the previous moments or step sizes (MBS)
        if self.optimizer == "sgd+":
            self.step_sizes = {param : 0 for param in self.learnable_params}
            self.bias_m = 0
        elif self.optimizer == "adam":
            self.moment1 = {param : 0 for param in self.learnable_params}
            self.moment2 = {param: 0 for param in self.learnable_params}
            self.bias_m1 = self.bias_m2 = 0
        ####################################################################

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)

            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in
                                        self.training_data[0]]  ## Associate label 0 with ecah sample
                self.class_1_samples = [(item, 1) for item in
                                        self.training_data[1]]  ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):
                cointoss = random.choice([0, 1])  ## When a batch is created by getbatch(), we want the
                ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)

            def getbatch(self):
                batch_data, batch_labels = [], []  ## First list for samples, the second for labels
                maxval = 0.0  ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval:
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item / maxval for item in batch_data]  ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0  ##  Average the loss over iterations for printing out
        ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids = self.forward_prop_one_neuron_model(data_tuples)  ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i])) ** 2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))  ##  Average the loss over batch
            avg_loss_over_literations += loss_avg
            if i % (self.display_loss_how_often) == 0:
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" % (i + 1, avg_loss_over_literations))  ## Display average loss
                avg_loss_over_literations = 0.0  ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg,
                                      [float(len(class_labels))] * len(class_labels)))
            ####################################################################
            # runs the backpropagation algorithm (I added iteration number as a parameter, MBS)
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg,
                                                             deriv_sigmoid_avg, iter=i+1)
            ####################################################################

        ####################################################################
        # I commented these plots to display them later (MBS)
        #plt.figure()
        #plt.plot(loss_running_record)
        #plt.show()
        return loss_running_record # returns the loss history (MBS)
        ####################################################################

    # backpropagation and update of one neuron classifier (I added iter as a parameter, MBS)
    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid, iter):
        input_vars = self.independent_vars
        vals_for_input_vars_dict = dict(zip(input_vars, vals_for_input_vars))
        vals_for_learnable_params = self.vals_for_learnable_params
        ################################################################################################################
        # All of this box is written by Mehmet Berk Sahin (update rule is taken from Prof. Kak's source code)
        # I did not change the variable names for easy comparison
        # do the step updates according to the optimizer
        if self.optimizer == "sgd+":
            # update loop for SGD+ (with momentum)
            for i, param in enumerate(self.vals_for_learnable_params):
                # calculate g_{t+1} parameter in the update rule eq. 2
                step = y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
                # do momentum weighted summation with the previous updates
                step += self.momentum * self.step_sizes[param]
                # save the latest step size (v_{t+1} parameter)
                self.step_sizes[param] = step
                # update the learnable parameter
                vals_for_learnable_params[param] += self.learning_rate * step
            # update the bias using the SGD+ optimizer
            self.bias_m = self.momentum * self.bias_m + y_error * deriv_sigmoid
            self.bias += self.learning_rate * self.bias_m
        elif self.optimizer == "adam":
            # update loop for Adam
            for i, param in enumerate(self.vals_for_learnable_params):
                # calculate g_{t+1} in the update rule eq. 3
                step = y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
                # do moment updates m_{t+1} and v_{t+1}, respectively
                self.moment1[param] = self.beta1 * self.moment1[param] + (1 - self.beta1) * step
                self.moment2[param] = self.beta2 * self.moment2[param] + (1 - self.beta2) * step ** 2
                # do bias correction for moments eq. 38 in Prof. Kak's Autograd slide
                m_unit = self.moment1[param] / (1 - self.beta1 ** iter)
                v_unit = self.moment2[param] / (1 - self.beta2 ** iter)
                # update the learnable parameters with bias-corrected moments
                self.vals_for_learnable_params[param] += self.learning_rate * m_unit / ((v_unit + 1e-8) ** 0.5)
            # calculate g_{t+1} in the update rule eq. 3 for bias term
            step = y_error * deriv_sigmoid
            # moment updates and bias corrections for the bias term
            self.bias_m1 = self.beta1 * self.bias_m1 + (1 - self.beta1) * step
            self.bias_m2 = self.beta2 * self.bias_m2 + (1 - self.beta2) * step ** 2
            m_unit = self.bias_m1 / (1 - self.beta1 ** iter)
            v_unit = self.bias_m2 / (1 - self.beta2 ** iter)
            # update the bias learnable parameter
            self.bias += self.learning_rate * m_unit / ((v_unit + 1e-8) ** 0.5)
        else:
            # Do normal SGD (This part is same as super class' SGD implementation)
            for i, param in enumerate(self.vals_for_learnable_params):
                step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
                self.vals_for_learnable_params[param] += step
            self.bias += self.learning_rate * y_error * deriv_sigmoid
        ################################################################################################################

    # training loop for multi neuron classifier
    def run_training_loop_multi_neuron_model(self, training_data):
        """
        DISCLAIMER: I copied this function from Prof. Kak's ComputationalGraphPrimer source code.
        I marked the adjustments I made by adding MBS (Mehmet Berk Sahin) at the end of the comments.
        And I pointed the modifications out by showing them in a box with '#' symbol.
        """

        ############################################################################
        # Initializes the necessary parameters for each learnable parameter (MBS)
        if self.optimizer == "sgd+":
            self.step_sizes = {param: 0 for param in self.learnable_params}
            self.bias_m = [0 for _ in range(self.num_layers - 1)]
        elif self.optimizer == "adam":
            self.moment1 = {param: 0 for param in self.learnable_params}
            self.moment2 = {param: 0 for param in self.learnable_params}
            self.bias_m1 = [0 for _ in range(self.num_layers - 1)]
            self.bias_m2 = [0 for _ in range(self.num_layers - 1)]
        ############################################################################

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)

            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """

            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in
                                        self.training_data[0]]  ## Associate label 0 with ecah sample
                self.class_1_samples = [(item, 1) for item in
                                        self.training_data[1]]  ## Associate label 1 with ecah sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):
                cointoss = random.choice([0, 1])  ## When a batch is created by getbatch(), we want the
                ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)

            def getbatch(self):
                batch_data, batch_labels = [], []  ## First list for samples, the second for labels
                maxval = 0.0  ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval:
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item / maxval for item in batch_data]  ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch

        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0, 1) for param in self.learnable_params}

        self.bias = [random.uniform(0, 1) for _ in
                     range(self.num_layers - 1)]  ## Adding the bias to each layer improves
        ##   class discrimination. We initialize it
        ##   to a random number.

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0  ##  Average the loss over iterations for printing out
        ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)  ## FORW PROP works by side-effect
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[
                self.num_layers - 1]  ## Predictions from FORW PROP
            y_preds = [item for sublist in predicted_labels_for_batch for item in
                       sublist]  ## Get numeric vals for predictions
            loss = sum([(abs(class_labels[i] - y_preds[i])) ** 2 for i in
                        range(len(class_labels))])  ## Calculate loss for batch
            loss_avg = loss / float(len(class_labels))  ## Average the loss over batch
            avg_loss_over_literations += loss_avg  ## Add to Average loss over iterations
            if i % (self.display_loss_how_often) == 0:
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" % (i + 1, avg_loss_over_literations))  ## Display avg loss
                avg_loss_over_literations = 0.0  ## Re-initialize avg-over-iterations loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            ############################################################################################################
            # runs backpropagation algorithm for multi neuron case (I added iter parameter, MBS)
            self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels, iter=i+1)
            ############################################################################################################
        ############################################################################
        # I commented these to diplay them later (MBS)
        #plt.figure()
        #plt.plot(loss_running_record)
        #plt.show()
        return loss_running_record # return the loss (MBS)
        ############################################################################

    # backpropagation and update of multi neuron classifier
    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels, iter):
        """
        DISCLAIMER: I copied this function from Prof. Kak's ComputationalGraphPrimer source code.
        I marked the adjustments I made by adding MBS (Mehmet Berk Sahin) at the end of the comments.
        And I pointed the modifications out by showing them in a box with '#' symbol.

        First note that loop index variable 'back_layer_index' starts with the index of
        the last layer.  For the 3-layer example shown for 'forward', back_layer_index
        starts with a value of 2, its next value is 1, and that's it.

        Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
        the samples in a batch.  To explain how this averaging is carried out by the
        backprop function, consider the last node on the example shown in the forward()
        function above.  Standing at the node, we look at the 'input' values stored in the
        variable "input_vals".  Assuming a batch size of 8, this will be list of
        lists. Each of the inner lists will have two values for the two nodes in the
        hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
        these values 'input vals' and store those in the variable "input_vals_avg".  Next we
        must carry out the same batch-based averaging for the partial derivatives stored in the
        variable "deriv_sigmoid".

        Pay attention to the variable 'vars_in_layer'.  These store the node variables in
        the current layer during backpropagation.  Since back_layer_index starts with a
        value of 2, the variable 'vars_in_layer' will have just the single node for the
        example shown for forward(). With respect to what is stored in vars_in_layer', the
        variables stored in 'input_vars_to_layer' correspond to the input layer with
        respect to the current layer.
        """
        # backproped prediction error:
        pred_err_backproped_at_layers = {i: [] for i in range(1, self.num_layers - 1)}
        pred_err_backproped_at_layers[self.num_layers - 1] = [y_error]
        for back_layer_index in reversed(range(1, self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[back_layer_index - 1]
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(
                map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
            deriv_sigmoid = self.gradient_vals_for_layers[back_layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg,
                                         [float(len(class_labels))] * len(class_labels)))
            vars_in_layer = self.layer_vars[back_layer_index]  ## a list like ['xo']
            vars_in_next_layer_back = self.layer_vars[back_layer_index - 1]  ## a list like ['xw', 'xz']

            layer_params = self.layer_params[back_layer_index]
            ## note that layer_params are stored in a dict like
            ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
            ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
            transposed_layer_params = list(zip(*layer_params))  ## creating a transpose of the link matrix

            backproped_error = [None] * len(vars_in_next_layer_back)
            for k, varr in enumerate(vars_in_next_layer_back):
                for j, var2 in enumerate(vars_in_layer):
                    backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] *
                                               pred_err_backproped_at_layers[back_layer_index][i]
                                               for i in range(len(vars_in_layer))])
            #                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
            pred_err_backproped_at_layers[back_layer_index - 1] = backproped_error
            input_vars_to_layer = self.layer_vars[back_layer_index - 1]
            for j, var in enumerate(vars_in_layer):
                layer_params = self.layer_params[back_layer_index][j]
                ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3
                ##  lecture slides for how the parameters are updated using the patial derivatives stored away
                ##  during forward propagation of data. The theory underlying these calculations is presented
                ##  in Slides 68 through 71.
            ############################################################################################################
                # I modified this section of the code (MBS)
                for i, param in enumerate(layer_params):
                    # calculate g_{t+1} in the update rule eq. 2
                    step = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j]
                    if self.optimizer == "sgd+":
                        # calculate the momentum weighted step size
                        self.step_sizes[param] = self.momentum * self.step_sizes[param] + step
                        # save the last step size, v_{t+1}
                        step = self.step_sizes[param]
                    elif self.optimizer == "adam":
                        # calculate and save moments m_{t+1} and v_{t+1}, respectively (eq. 3 in hw)
                        self.moment1[param] = self.beta1 * self.moment1[param] + (1 - self.beta1) * step
                        self.moment2[param] = self.beta2 * self.moment2[param] + (1 - self.beta2) * step ** 2
                        # do bias corrections
                        m_unit = self.moment1[param] / (1 - self.beta1 ** iter)
                        v_unit = self.moment2[param] / (1 - self.beta2 ** iter)
                        # calculate the bias-corrected step size
                        step = m_unit / ((v_unit + 1e-8) ** 0.5)
                    # update the learnable parameter according to the given optimizer
                    self.vals_for_learnable_params[param] += self.learning_rate * step
            # calculate g_{t+1} in update rules given in hw3
            step = sum(pred_err_backproped_at_layers[back_layer_index]) * sum(deriv_sigmoid_avg) / len(deriv_sigmoid_avg)
            # choose the update rule according to the optimizer name
            if self.optimizer == "sgd+":
                # calculate the new step size (v_{t+1}) by momentum weighted summation
                prev_bias = self.bias_m[back_layer_index - 1] # v_t
                self.bias_m[back_layer_index - 1] = self.momentum * prev_bias + step
                step = self.bias_m[back_layer_index - 1] # v_{t+1}
            elif self.optimizer == "adam":
                # calculate and save moments m_{t+1} and v_{t+1} for biases, respectively (eq. 3 in hw)
                self.bias_m1[back_layer_index - 1] = self.beta1 * self.bias_m1[back_layer_index - 1] + (1 - self.beta1) * step
                self.bias_m2[back_layer_index - 1] = self.beta2 * self.bias_m2[back_layer_index - 1] + (1 - self.beta2) * step ** 2
                # do bias corrections
                m_unit = self.bias_m1[back_layer_index - 1] / (1 - self.beta1 ** iter)
                v_unit = self.bias_m2[back_layer_index - 1] / (1 - self.beta2 ** iter)
                # calculate the bias corrected step size
                step = m_unit / ((v_unit + 1e-8) ** 0.5)
            # do update according to the given optimizer and resulted step size
            self.bias[back_layer_index - 1] += self.learning_rate * step
            ############################################################################################################
# compares SGD, SGD+ and Adam with one neuron classifier
def one_neuron_experiment(learning_rate=5e-3):
    # for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # SGD
    sgd_model = MyComputationalGraphPrimer(
        one_neuron_model=True,
        expressions=['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars=['xw'],
        dataset_size=5000,
        learning_rate=learning_rate,
        #               learning_rate = 5 * 1e-2,
        training_iterations=40000,
        batch_size=8,
        display_loss_how_often=100,
        debug=True,
        optimizer="sgd"
    )
    # SGD+
    sgdp_model = MyComputationalGraphPrimer(
        one_neuron_model=True,
        expressions=['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars=['xw'],
        dataset_size=5000,
        learning_rate=learning_rate,
        #               learning_rate = 5 * 1e-2,
        training_iterations=40000,
        batch_size=8,
        display_loss_how_often=100,
        debug=True,
        optimizer="sgd+",
        momentum=0.9
    )
    # ADAM
    adam_model = MyComputationalGraphPrimer(
        one_neuron_model=True,
        expressions=['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars=['xw'],
        dataset_size=5000,
        learning_rate=learning_rate,
        #               learning_rate = 5 * 1e-2,
        training_iterations=40000,
        batch_size=8,
        display_loss_how_often=100,
        debug=True,
        optimizer="adam",
        beta1=0.9,
        beta2=0.99
    )

    # set the parameters
    sgd_model.parse_expressions()
    sgdp_model.parse_expressions()
    adam_model.parse_expressions()
    # generate a shared training data
    training_data = sgd_model.gen_training_data()
    # start training of each model
    loss_sgd_cgp = sgd_model.run_training_loop_one_neuron_model(training_data)
    print("Training for SGD with single neuron is completed.")
    loss_sgdp_cgp = sgdp_model.run_training_loop_one_neuron_model(training_data)
    print("Training for SGD+ with single neuron is completed.")
    loss_adam_cgp = adam_model.run_training_loop_one_neuron_model(training_data)
    print("Training for ADAM with single neuron is completed.")
    # plot the training histories
    plt.figure()
    plt.plot(loss_sgd_cgp)
    plt.plot(loss_sgdp_cgp)
    plt.plot(loss_adam_cgp)
    plt.legend(["SGD Training Loss", "SGD+ Training Loss", "Adam Training Loss"])
    plt.title("Loss v.s. Step (Single-Neuron Model)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

# compares SGD, SGD+ and Adam with multi neuron classifier
def multi_neuron_experiment(learning_rate=5e-3):
    # for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # SGD for multi-neuron classifier
    sgd_model = MyComputationalGraphPrimer(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learning_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
    )
    # SGD+ for multi-neuron classifier
    sgdp_model = MyComputationalGraphPrimer(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learning_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
               optimizer='sgd+',
               momentum=0.9
    )
    # ADAM for multi-neuron classifier
    adam_model = MyComputationalGraphPrimer(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learning_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
               optimizer='adam',
               beta1=0.9,
               beta2=0.99
    )

    # setup
    sgd_model.parse_multi_layer_expressions()
    sgdp_model.parse_multi_layer_expressions()
    adam_model.parse_multi_layer_expressions()
    # generate data
    training_data = sgd_model.gen_training_data()
    # start learning
    loss_sgd_cgp = sgd_model.run_training_loop_multi_neuron_model(training_data)
    print("Training for SGD with multi neuron is completed.")
    loss_sgdp_cgp = sgdp_model.run_training_loop_multi_neuron_model(training_data)
    print("Training for SGD+ with multi neuron is completed.")
    loss_adam_cgp = adam_model.run_training_loop_multi_neuron_model(training_data)
    print("Training for ADAM with multi neuron is completed.")
    # plot the training histories for multi neuron classifier
    plt.figure()
    plt.plot(loss_sgd_cgp)
    plt.plot(loss_sgdp_cgp)
    plt.plot(loss_adam_cgp)
    plt.legend(["SGD Training Loss", "SGD+ Training Loss", "Adam Training Loss"])
    plt.title("Loss v.s. Step (Multi-Neuron Model)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

# main code
if __name__ == "__main__":
    # for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # experiments for single-neuron case
    print("#"*30)
    print(f"The experiment for single-neuron classifier with learning rate 1e-3 is being started...")
    one_neuron_experiment(learning_rate=1e-3)
    print("The experiment for single-neuron class with learning rate 1e-3 was ended.")
    print(f"The experiment for single-neuron classifier with learning rate 3e-3 is being started...")
    one_neuron_experiment(learning_rate=3e-3)
    print("The experiment for single-neuron class with learning rate 3e-3 was ended.")
    # experiments for multi-neuron case
    print("#" * 30)
    print("The experiment for multi-neuron classifier with learning rate 1e-3 is being started...")
    multi_neuron_experiment(learning_rate=1e-3)
    print("The experiment for multi-neuron class with learning rate 1e-3 was ended.")
    print("The experiment for multi-neuron classifier with learning rate 3e-3 is being started...")
    multi_neuron_experiment(learning_rate=3e-3)
    print("The experiment for multi-neuron class with learning rate 3e-3 was ended.")

    print("Homework 3 code come to an end. ")

