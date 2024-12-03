
library(tidyverse)
library(sits)
library(sitsdata)

library(torch)


#df <- samples_matogrosso_mod13q1 #Okay(model$forward())
#df <- samples_deforestation      #Okay(model$forward())
#df <- samples_prodes_4classes    #Okay(model$forward())
#df <- samples_cerrado_cbers      #Okay(model$forward())
df <- samples_cerrado_mod13q1    #Okay(model$forward())

n_samples <- nrow(df)
n_times <- nrow(df[1, ]$time_series[[1]])
n_bands <- length(setdiff(names(df[["time_series"]][[1]]), "Index"))

# Get samples' labels(classes)
labels <- sort(unique(df[["label"]]), na.last = TRUE)
# Create numeric labels vector
code_labels <- seq_along(labels)
names(code_labels) <- labels



#PART 1: Building the Predictors from data
#-----------------------------------------
# 1.a- Getting the time series
# Loose version of the .ts() sits package internal function
get_ts <- function(x) {
    # Columns every ts needs to have
    ts_cols <- c("sample_id", "label")
    # Create the sample_id column
    x[["sample_id"]] <- seq_along(x[["time_series"]])
    # Extract the time series from time_series column
    ts <- tidyr::unnest(
        data = x[c(ts_cols, "time_series")],
        cols = "time_series"
    )
    # Return time series
    ts
}
# 1.b- Creating the predictors
# Loose version of the .predictors() sits package internal function
make_predictors <- function(samples) {
    # Columns every predictor needs to have
    pred_cols <- c("sample_id", "label")
    # Bands of the first sample governs whole samples data
    bands <- setdiff(names(samples[["time_series"]][[1]]), "Index")
    # Creating the predictors
    pred <- get_ts(samples)
    pred <- pred[c(pred_cols, bands)]
    # Add sequence 'index' column grouped by 'sample_id'
    # The new variable index, an int, is different from the existing Index, a date
    pred <- pred |>
        dplyr::select("sample_id", "label", dplyr::all_of(bands)) |>
        dplyr::group_by(.data[["sample_id"]]) |>
        dplyr::mutate(index = seq_len(dplyr::n())) |>
        dplyr::ungroup()
    # Rearrange data to create predictors
    pred <- tidyr::pivot_wider(
        data = pred, names_from = "index", values_from = dplyr::all_of(bands),
        names_prefix = if (length(bands) == 1) bands else "",
        names_sep = ""
    )
    # Return predictors
    pred
}

D <- make_predictors(df)
X <- D |>
    dplyr::select(!sample_id:label)

# PART 2: Getting the predictors and labels ready for torch
#---------------------------------------------------------
# Reshaping predictors to use tempCNN

split_data <- function(X, y, validation_split) {
    # Get the number of samples
    num_samples <- X$size(1)  # Assuming the first dimension is the batch size

    # Calculate the split index
    split_index <- as.integer(num_samples * (1 - validation_split))

    # Shuffle the indices
    indices <- torch_randperm(num_samples) + 1

    # Create training and testing indices
    train_indices <- torch_tensor(as.array(indices[1:split_index]), dtype = torch_long())
    test_indices <- torch_tensor(as.array(indices[(split_index + 1):num_samples]), dtype = torch_long())

    # Split the data
    x_train <- X[train_indices, ..]
    y_train <- y[train_indices, ..]
    x_test <- X[test_indices, ..]
    y_test <- y[test_indices, ..]

    return(list(
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test
    ))
}

X <- array(
    data = as.matrix(X),
    dim = c(n_samples, n_times, n_bands)
)

Y <- unname(code_labels[D[["label"]]])

X <- torch_tensor(X)
y <- torch_tensor(Y)

validation_split <- 0.2

data_split <- split_data(X, y, validation_split)

x_train <- data_split$x_train
y_train <- data_split$y_train
x_test <- data_split$x_test
y_test <- data_split$y_test




# PART 3: Defining the network's architecture
#--------------------------------------------
## module to be replaced by sits internal function .torch_conv1d_batch_norm_relu_dropout()
fcn_block <- torch::nn_module(
    classname = "fcn_block",
    initialize = function(input_dim,
                          output_dim,
                          kernel_size,
                          padding,
                          dropout_rate) {
        self$block <- torch::nn_sequential(
            torch::nn_conv1d(
                in_channels = input_dim,
                out_channels = output_dim,
                kernel_size = kernel_size,
                padding = padding
            ),
            torch::nn_batch_norm1d(num_features = output_dim),
            torch::nn_relu(),
            torch::nn_dropout(p = dropout_rate)
        )
    },
    forward = function(x) {
        # Input shape for debugging
        cat("fcn_block - Input shape:", x$shape, "\n")
        x <- self$block(x)
        # Output shape for debugging
        cat("fcn_block - Output shape:", x$shape, "\n")
        x
    }
)

# The LSTM/FCN for time series:
sits_lstm_fcn <- torch::nn_module(
    classname = "model_lstm_fcn",
    initialize = function(n_bands,
                          n_times,
                          n_labels,
                          kernel_sizes,
                          hidden_dims,
                          lstm_cells,
                          hidden_size,
                          dropout_rates) {
        # Upper branch: LSTM with dimension shift
        self$lstm <- torch::nn_lstm(
            input_size = n_times,
            hidden_size = hidden_size,
            dropout = dropout_rates[[1]],
            num_layers = lstm_cells,
            batch_first = TRUE
        )
        # Lower branch: Fully Convolutional Layers and avg pooling
        self$conv_bn_relu1 <- fcn_block(
            input_dim = n_bands,
            output_dim = hidden_dims[[1]],
            kernel_size = kernel_sizes[[1]],
            padding = as.integer(kernel_sizes[[1]] %/% 2),
            dropout_rate = dropout_rates[[1]]
        )
        self$conv_bn_relu2 <- fcn_block(
            input_dim = hidden_dims[[1]],
            output_dim = hidden_dims[[2]],
            kernel_size = kernel_sizes[[2]],
            padding = as.integer(kernel_sizes[[2]] %/% 2),
            dropout_rate = dropout_rates[[2]]
        )
        self$conv_bn_relu3 <- fcn_block(
            input_dim = hidden_dims[[2]],
            output_dim = hidden_dims[[3]],
            kernel_size = kernel_sizes[[3]],
            padding = as.integer(kernel_sizes[[3]] %/% 2),
            dropout_rate = dropout_rates[[3]]
        )
        # Global average pooling
        self$pooling <- torch::nn_adaptive_avg_pool1d(output_size = hidden_size)
        # Flattening 3D tensor to run the dense layer
        self$flatten <- torch::nn_flatten()
        # Final module: dense layer outputting the number of labels
        self$dense <- torch::nn_linear(
            in_features = n_bands * hidden_size * 2,
            out_features = n_labels
        )
    },
    forward = function(x) {
        # LSTM input shape for debugging
        cat("sits_lstm_fcn - LSTM input shape:", x$shape, "\n")
        # dimension shift and LSTM forward pass
        x_lstm <- x$permute(c(1, 3, 2)) |>
            self$lstm()
        #x_lstm <- self$lstm(x)
        # LSTM output shape for debugging
        cat("sits_lstm_fcn - LSTM output shape:", x_lstm[[1]]$shape, "\n")
        # FCN forward pass
        # FCN input shape for debugging
        cat("sits_lstm_fcn - FCN input shape:", x$shape, "\n")
        x_fcn <- x$permute(c(1, 3, 2)) |>
            self$conv_bn_relu1() |>
            self$conv_bn_relu2() |>
            self$conv_bn_relu3() |>
            self$pooling()
        # FCN output shape for debugging
        cat("Pooling output shape:", x_fcn$shape, "\n")
        cat("LSTM output shape:", x_lstm[[1]]$shape, "\n")
        # Concatenate upper and lower branches
        x_combined <- torch_cat(list(x_lstm[[1]], x_fcn), dim = 2)
        cat("sits_lstm_fcn - Combined output shape:", x_combined$shape, "\n")
        x_flat <- self$flatten(x_combined)
        cat("sits_lstm_fcn - Flattened output shape:", x_flat$shape, "\n")
        cat("sits_lstm_fcn - Expected dense input:",  n_bands * hidden_size * 2, "\n")
        x_out <- self$dense(x_flat)
    }
)

# Defining model parameters for debugging


n_bands = n_bands
n_times = n_times
n_labels = length(labels)
kernel_sizes = list(3, 3, 3)
hidden_dims = list(16, 32, n_bands)
lstm_cells = 1
hidden_size = 8
dropout_rates = list(0.0, 0.2, 0.3)
optimizer = torch::optim_adamw
opt_hparams = list(
    lr = 5.0e-04,
    eps = 1.0e-08,
    weight_decay = 1.0e-06
)
lr_decay_epochs = 1
lr_decay_rate = 0.95
patience = 20
min_delta = 0.01
verbose = FALSE
batch_size = 64

# Training the model with luz package

# Initialize the model
model <- sits_lstm_fcn(
    n_bands = n_bands,
    n_times = n_times,
    n_labels = n_labels,
    kernel_sizes = kernel_sizes,
    hidden_dims = hidden_dims,
    lstm_cells = lstm_cells,
    hidden_size = hidden_size,
    dropout_rates = dropout_rates
)

# Forward pass through the model
output <- model(x_train)


torch_model <- luz::setup(
    # How do we use class name?
    module = sits_lstm_fcn,
    loss = torch::nn_cross_entropy_loss(),
    metrics = list(luz::luz_metric_accuracy()),
    optimizer = optimizer
    ) |>
    luz::set_opt_hparams(
        ## FIND THIS: !!!optim_params_function
        lr = 5.0e-04,
        eps = 1.0e-08,
        weight_decay = 1.0e-06
    ) |>
    luz::set_hparams(
        n_bands = n_bands,
        n_times = n_times,
        n_labels = n_labels,
        lstm_cells = lstm_cells,
        hidden_size = hidden_size,
        kernel_sizes = kernel_sizes,
        hidden_dims = hidden_dims,
        dropout_rates = dropout_rates
    ) |>
    luz::fit(
        data = list(x_train, y_train),
        epochs = epochs,
        valid_data = list(x_test, y_test),
        callbacks = list(
            luz::luz_callback_early_stopping(
                monitor = "valid_loss",
                patience = patience,
                min_delta = min_delta,
                mode = "min"
            ),
            luz::luz_callback_lr_scheduler(
                torch::lr_step,
                step_size = lr_decay_epochs,
                gamma = lr_decay_rate
            )
        ),
        #accelerator = luz::accelerator(cpu = cpu_train),
        dataloader_options = list(batch_size = batch_size),
        verbose = verbose
    )
