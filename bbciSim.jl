# Simulating Neural Connectivity and Entropic Modulation: A BBCI Framework for Enhancing Hemispheric Integration in Autism Spectrum Disorder

# Required libraries for project
using Pkg
Pkg.add(["Plots", "Flux", "Random", "DataFrames", "CSV"])

### Simulate Neural Activity with Hemispheric Isolation ###
# Creating a simulation of neural activity for two brain regions, introducing hemispheric isolation by reducing inter-regional connectivity.

using Random
using Plots

# Define parameters for simulation
n_time_steps = 100  # Number of time steps
n_neurons = 10      # Number of neurons in each brain region
region1_activity = zeros(Float64, n_time_steps, n_neurons)  # Region 1
region2_activity = zeros(Float64, n_time_steps, n_neurons)  # Region 2

# Simulate isolated neural activity with low correlation between regions
for t in 2:n_time_steps
    region1_activity[t, :] .= region1_activity[t - 1, :] + randn(n_neurons) * 0.1  # Random activity with some temporal correlation
    region2_activity[t, :] .= region2_activity[t - 1, :] + randn(n_neurons) * 0.05  # Reduced correlation in Region 2
end

# Plot the simulated activity
plot(1:n_time_steps, region1_activity[:, 1], label="Neuron 1 in Region 1", ylabel="Firing Rate", xlabel="Time")
plot!(1:n_time_steps, region2_activity[:, 1], label="Neuron 1 in Region 2")

### Incorporate Neural Entropy (Psychedelics Effect) ###
# Add a parameter to simulate increased neural entropy, representing enhanced brain malleability.

function apply_entropy!(activity, entropy_level)
    for t in 2:n_time_steps
        activity[t, :] .= activity[t - 1, :] + randn(n_neurons) * entropy_level  # Increase randomness in firing rates
    end
end

# Apply entropy to simulate the effect of psychedelics
entropy_level = 0.3  # Adjust this value to simulate the level of enhanced neural malleability
apply_entropy!(region2_activity, entropy_level)

# Plot the activity after applying entropy
plot(1:n_time_steps, region2_activity[:, 1], label="Neuron 1 in Region 2 (Post-Entropy)", ylabel="Firing Rate", xlabel="Time")

### Represent Intent Recognition Using a Simple Machine Learning Model ###
# Use a basic feedforward neural network to classify intent based on neural activity.
# We’ll also add a feature to measure inter-regional connectivity.

using Flux

# Generate synthetic data for intent classification
X = vcat(region1_activity, region2_activity)  # Combine activity from both regions
intent_labels = vcat(ones(Int64, n_time_steps), 2 * ones(Int64, n_time_steps))  # Label: 1 for Region 1, 2 for Region 2

# Normalize data
X_normalized = X ./ maximum(abs.(X))

# Reshape data to match the expected input size for the neural network
X_reshaped = reshape(X_normalized, n_neurons, 2 * n_time_steps)

# Convert data to Float32 to match model parameters
X_reshaped = Float32.(X_reshaped)

# Define a simple neural network model
model = Chain(
    Dense(n_neurons, 16, relu),  # First layer with 16 hidden units
    Dense(16, 2),                # Output layer (2 classes: intent 1 and 2)
    softmax                      # Softmax activation for classification
)

# Define loss function and optimizer
loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)
optimizer = Descent(0.01)  # Gradient descent optimizer

# Convert data to tensors
X_tensor = X_reshaped
Y_tensor = Flux.onehotbatch(intent_labels, 1:2)

# Initialize optimizer state
opt_state = Flux.setup(optimizer, model)

# Train the model
for epoch in 1:100
    grads = Flux.gradient(Flux.trainable(model)) do
        loss_fn(X_tensor, Y_tensor)
    end
    Flux.Optimise.update!(optimizer, opt_state, grads)
end

println("Model training complete.")

### Modulate Neural Activity and Enhance Connectivity ###
# Simulate the BBCI guiding neural activity to address hemispheric isolation.

# Define a stimulation function that encourages synchronization between regions
guided_activity = copy(region2_activity)

function stimulate_and_sync!(activity1, activity2, threshold)
    for t in 1:n_time_steps
        for n in 1:n_neurons
            # Stimulate underactive neurons in Region 2
            if activity2[t, n] < threshold
                activity2[t, n] += 0.5  # Increase activity for underactive neurons
            end
            # Encourage synchronization with Region 1
            activity2[t, n] += 0.1 * (activity1[t, n] - activity2[t, n])  # Adjust towards Region 1
        end
    end
end

stimulate_and_sync!(region1_activity, guided_activity, 0.3)

# Plot before and after stimulation
plot(1:n_time_steps, region2_activity[:, 1], label="Original Activity")
plot!(1:n_time_steps, guided_activity[:, 1], label="Stimulated Activity", title="Effect of BBCI on Hemispheric Isolation")

### Visualize Results ###
# Combine original and guided activity into a DataFrame and save as CSV.
using DataFrames, CSV

data = DataFrame(
    Time = 1:n_time_steps,
    OriginalActivity = region2_activity[:, 1],
    StimulatedActivity = guided_activity[:, 1]
)

CSV.write("neural_activity.csv", data)
println("Results saved to neural_activity.csv.")

# Display the final guided activity plot.
plot(data.Time, data.OriginalActivity, label="Original Activity", xlabel="Time", ylabel="Firing Rate", title="Neural Stimulation")
plot!(data.Time, data.StimulatedActivity, label="Stimulated Activity")