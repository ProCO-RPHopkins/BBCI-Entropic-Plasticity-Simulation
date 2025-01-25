# Simplified model to simulate how a bidirectional brain-computer interface (BBCI) might interpret and guide neural activity, incorporating the concept of entropic brian theory to enhance neuroplasticity.

# Key competencies
#   1. Neuroscience - Simulating neural activity and understanding neuroplasticity
#   2. Entropic Brain Theory - Using entropy to enhance neural malleability and plasticity
#   3. Brain-Computer Interface - Recognizing intent and guiding neural activity through stimulation
#   4. Data Science - Collecting, processing, and visualizing neural data
#   5. Machine Learning - Using AI models to decode neural intent and classify neural activity patterns
#   6. Coding - Implementing the model in Julia within a Jupyter Notebook
#   7. Modeling - Simulating synaptogensis, neural plasticity, and activity modulation

# Problem Statement
    # Autism spectrum disorder (ASD) is often associated with atypical neural activity patterns, including hemispheric isolation and underdeveloped neural connectivity, which can lead to challenges in motor planning, social communication, and sensory processing. While traditional therapies focus on behavioral interventions, they often fail to address underlying neurological deficiencies directly. The development of Bidirectional Brain-Computer Interfaces (BBCIs), combined with advancements in computational neuroscience and AI, offers a promising solution. By recognizing intent and guiding neural activity through external stimulation, BBCIs can potentially facilitate neuroplasticity and create new neural pathways.

    # However, current models lack integration with frameworks that enhance neuroplasticity further, such as those inspired by the entropic brain hypothesis related to psychedelics. Increased neural entropy may make the brain more receptive to rewiring, yet this aspect remains underexplored in BBCI systems.

     # How can we design a simplified model to simulate a bidirectional brain-computer interface (BBCI) that can interpret neural activity and guide it to enhance neuroplasticity?
 
# Objectives
    # This project aims to simulate how BBCIs, augmented by AI and theoretical psychedelics enhancements to neuroplasticity, can be used to recognize intent, stimulate appropriate brain regions, and foster long-term functional connectivity. By addressing this problem, we can advance the theoretical groundwork for innovative therapies that combine neuroscience, machine learning, and neurotechnology to improve outcomes for individuals with ASD or similar neurodevelopmental conditions.

    # In the context of a BBCI, we are interested in understanding how neural activity can be interpreted and guided to enhance neuroplasticity. We aim to develop a simplified model that can simulate the behavior of the brain-computer interface and incorporate the concept of entropic brain theory to enhance neural malleability. The model should be able to decode neural activity patterns, recognize intent, and guide neural activity through stimulation to promote neuroplasticity. By simulating the interactions between neural activity, stimulation, and plasticity, we can explore how a bidirectional brain-computer interface might function and how it could be used to enhance cognitive abilities and learning processes.

# Methods
    # 1. Simulation of Neural Activity - Neural activity was simulated as a time series representing the firing rates of neurons in two brain regions. Each neuron’s activity was modeled with temporal correlation, introducing variability to reflect stochastic neuronal firing. The activity was initialized as a zero matrix and evolved over time using random perturbations:
        # Parameters - 100 time steps, 10 neurons per region.
        # Output - Two datasets representing neural activity for Region 1 and Region 2.

    # 2. Enhanced Neural Malleability Using Entropy - To simulate the effects of increased neural plasticity (e.g., influenced by psychedelics or entropic brain states), a custom entropy function was applied to the simulated activity. The function amplified random perturbations, increasing variability in the firing rates:
        # Entropy Level - Adjustable parameter (default set to 0.3).
        # Purpose - Mimicked the neural malleability required for synaptic rewiring and behavioral modulation.

    # 3. Intent Recognition with Machine Learning - A feedforward neural network was developed using the Flux.jl library to classify intent based on neural activity:
        # Data Preparation: Neural activity datasets from both regions were combined and labeled manually (Region 1 as Intent 1; Region 2 as Intent 2).
        # Model Architecture:
            # Input Layer - 10 neurons.
            # Hidden Layer - 16 neurons with ReLU activation.
            # Output Layer - 2 neurons with softmax activation for classification.
        # Training Parameters:
            # Loss Function - Logit cross-entropy.
            # Optimizer - Gradient Descent (learning rate = 0.01).
            # Epochs - 100 iterations.
        # Output: A trained model capable of decoding neural intent based on input activity.

    # 4. Neural Modulation via Stimulation - To demonstrate bidirectional control, a stimulation function was implemented to modulate neural activity. The function identified underactive neurons and increased their firing rates above a defined threshold:
        # Threshold - Activity values below 0.3 were selectively augmented.
        # Mechanism - Stimulated neurons experienced a 0.5 increase in activity to guide them toward optimal firing rates.

    # 5. Visualization and Data Analysis - The simulated and modulated neural activity datasets were visualized and analyzed:
        # Plots: Time-series plots comparing original and stimulated neural activity.
        # Data Export: Results saved as a CSV file for further analysis using the DataFrames.jl and CSV.jl libraries.

    # 6. Implementation Environment - The project was implemented in Julia using Jupyter Notebook. The following libraries were employed:
        # Plots.jl: Visualization of neural activity.
        # Flux.jl: Machine learning model development.
        # Random.jl: Neural activity simulation.
        # DataFrames.jl and CSV.jl: Data handling and export.


# Results
    # The model successfully simulated neural activity in two brain regions, incorporating temporal correlation and stochastic variability. The application of an entropy function increased neural malleability, enhancing the potential for synaptic rewiring and plasticity. The machine learning model accurately classified neural intent based on the combined activity patterns from both regions, demonstrating the feasibility of decoding user intent from neural data.

    # Stimulation of underactive neurons through the bidirectional interface effectively modulated neural activity, guiding the system toward desired states. The visualization of original and stimulated neural activity highlighted the impact of stimulation on the firing rates of individual neurons, illustrating the potential for targeted neural modulation.

    # The project showcased the integration of neuroscience, machine learning, and computational modeling to simulate a bidirectional brain-computer interface capable of interpreting neural activity and guiding it to enhance neuroplasticity. By combining theoretical frameworks such as entropic brain theory with practical applications in neural modulation, the model laid the foundation for future research in neurotechnology and cognitive enhancement.