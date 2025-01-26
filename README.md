# Simulating Neural Connectivity and Entropic Modulation: A BBCI Framework for Enhancing Hemispheric Integration in Autism Spectrum Disorder

## Overview

    Simplified model to simulate how a bidirectional brain-computer interface (BBCI)
    might interpret and guide neural activity, incorporating the concept of entropic
    brain theory to enhance neuroplasticity. This model aims to address challenges
    associated with autism spectrum disorder (ASD), which often involves atypical
    neural activity patterns such as hemispheric isolation and underdeveloped neural
    connectivity.

## Key competencies

    1. Neuroscience - Simulating neural activity and understanding neuroplasticity
    2. Entropic Brain Theory - Using entropy to enhance neural malleability and plasticity
    3. Bidirectional Brain-Computer Interface (BBCI)- Recognizing intent and guiding neural activity through stimulation
    4. Data Science - Collecting, processing, and visualizing neural data
    5. Statistics - Analyzing neural activity, measuring connectivity, and validating model outcomes
    6. Machine Learning - Using AI models to decode neural intent, classify activity patterns, and optimize performance
    7. Artificial Intelligence (AI) - Implementing intent recognition, activity modulation, and adaptive learning algorithms
    8. Coding - Developing and implementing the model in Julia within a Jupyter Notebook
    9. Modeling - Simulating synaptogensis, neural plasticity, and activity modulation
    10. Computational Neuroscience - Integrating theoretical concepts with practical applications in neural interfacing
    11. Systems Neuroscience - Investigating and simulating inter-regional connectivity and hemispheric synchronization

## Problem Statement

    Autism spectrum disorder (ASD) is often associated with atypical neural activity patterns, including hemispheric isolation
    and underdeveloped neural connectivity, which can lead to challenges in motor planning, social communication, and sensory
    processing. While traditional therapies focus on behavioral interventions, they often fail to address underlying neurological
    deficiencies directly. The development of Bidirectional Brain-Computer Interfaces (BBCIs), combined with advancements in 
    computational neuroscience and AI, offers a promising solution. By recognizing intent and guiding neural activity through
    external stimulation, BBCIs can potentially facilitate neuroplasticity and create new neural pathways.

    However, current models lack integration with frameworks that enhance neuroplasticity further, such as those inspired
    by the entropic brain hypothesis related to psychedelics. Increased neural entropy may make the brain more receptive to rewiring,
    yet this aspect remains underexplored in BBCI systems.

    How can we design a simplified model to simulate a bidirectional brain-computer interface (BBCI) that can interpret neural
    activity and guide it to enhance neuroplasticity?

## Objectives

    This project aims to simulate how BBCIs, augmented by AI and theoretical psychedelics enhancements to neuroplasticity,
    can be used to recognize intent, stimulate appropriate brain regions to create or enhance inter-hemispheric and inter-regional
    connectivity, and foster long-term functional connectivity. By addressing this problem, we can advance the theoretical
    groundwork for innovative therapies that combine neuroscience, machine learning, and neurotechnology to improve outcomes for
    individuals with ASD or similar neurodevelopmental conditions.

    In the context of a BBCI, we are interested in understanding how neural activity can be interpreted and guided to enhance 
    neuroplasticity. We aim to develop a simplified model that can simulate the behavior of the brain-computer interface and incorporate
    the concept of entropic brain theory to enhance neural malleability. The model should be able to decode neural activity 
    patterns, recognize intent, and guide neural activity through stimulation to promote neuroplasticity. By simulating the 
    interactions between neural activity, stimulation, and plasticity, we can explore how a bidirectional brain-computer interface
    might function and how it could be used to enhance cognitive abilities and learning processes.

## Methods

    1. Simulation of Neural Activity - Neural activity was simulated as a time series representing the firing rates of neurons in
    two brain regions. Hemispheric isolation was introduced by reducing the correlation between regions while maintaining temporal
    correlation within each region. The activity was initialized as a zero matrix and evolved over time using random perturbations:
        * Parameters - 100 time steps, 10 neurons per region.
        * Output - Two datasets representing neural activity for Region 1 and Region 2,  with weak connectivity between regions
        to model hemispheric isolation.

    2. Enhanced Neural Malleability Using Entropy - To simulate the effects of increased neural plasticity (e.g., influenced by
    psychedelics or entropic brain states), a custom entropy function was applied to the neural activity of Region 2. The function
    amplified random perturbations, introducing variability in the firing rates:
        * Entropy Level - Adjustable parameter (default set to 0.3).
        * Purpose - Modeled the enhanced neural malleability necessary for synaptic rewiring, aligning with the entropic brain
        theory.

    3. Intent Recognition with Machine Learning - A feedforward neural network was trained to classify intent based on the neural
    activity of both regions. Inter-regional activity patterns were combined to assess connectivity and train the model:
        * Data Preparation: A feedforward neural network was trained to classify intent based on the neural activity of both regions.
        Inter-regional activity patterns were combined to assess connectivity and train the model:
        * Model Architecture:
            * Input Layer - 10 neurons (representing neural firing rates).
            * Hidden Layer - 16 neurons with ReLU activation.
            * Output Layer - 2 neurons with softmax activation for classification.
        * Training Parameters:
            * Loss Function - Logit cross-entropy.
            * Optimizer - Gradient Descent (learning rate = 0.01).
            * Epochs - 100 iterations.
        * Output: A trained model capable of decoding neural intent based on input activity.

    4. Neural Modulation via Stimulation and Synchronization - To demonstrate bidirectional control, a stimulation function was
    implemented to guide neural activity and reduce hemispheric isolation by promoting synchronization between regions:
        * Stimulation of underactive neurons - Neurons in Region 2 with firing rates below a threshold (0.3) were selectively
        augmented by adding a 0.5 increase to their activity.
        * Synchronization - Neural activity in Region 2 was dynamically adjusted to move closer to the firing patterns of Region 1. This
        step simulated the effects of BBCI-driven inter-regional connectivity.
        * Output - Synchronized datasets for Region 1 and Region 2, representing enhanced inter-hemispheric communication.

    5. Visualization and Data Analysis - The neural activity datasets were visualized and analyzed to illustrate changes resulting
    from entropy application and stimulation:
        * Plots:
            * Time-series plots compared original neural activity to activity after entropy amplification and stimulation.
            * Visualization of synchronization effects across regions.
        * Data Export: Results were saved as a CSV file for further analysis using the DataFrames.jl and CSV.jl libraries.

    6. Implementation Environment - The project was implemented in Julia using Jupyter Notebook. The following libraries were 
    employed:
        * Plots.jl: Visualization of neural activity.
        * Flux.jl: Machine learning model development.
        * Random.jl: Neural activity simulation.
        * DataFrames.jl and CSV.jl: Data handling and export.

## Results

    Neural Activity Simulation - The model successfully simulated neural activity in two brain regions, introducing hemispheric
    isolation by reducing inter-regional connectivity. The activity patterns incorporated temporal correlations and
    stochastic variability, accurately reflecting atypical neural activity associated with ASD.

    Entropy-Driven Neural Malleability - The application of the entropy function enhanced neural malleability, 
    increasing variability in firing rates. This simulated the effects of neuroplasticity enhancement, aligning
    with theoretical frameworks such as entropic brain theory.

    Intent Recognition with Machine Learning - The feedforward neural network classified intent based on the combined activity
    patterns from both brain regions. The model achieved accurate classification of user intent, validating its ability to decode
    neural activity effectively.

    Neural Modulation via Simulation - The bidirectional brain-computer interface (BBCI) successfully modulated neural activity by
    stimulating underactive neurons. The stimulation process reduced hemispheric isolation by encouraging synchronization
    between brain regions and guiding activity toward optimal states.

    Visualization and Impact - Visualizations demonstrated the effect of entropy on neural malleability and the impact of
    stimulation on firing rates. The guided activity plots highlighted the potential of targeted neural modulation for improving
    inter-regional connectivity and addressing atypical activity patterns.

    Integrated Approach and Implications - This project showcased a novel integration of computational neuroscience, machine
    learning, and theoretical frameworks to simulate a BBCI. By combining advanced modeling techniques with practical
    applications in neural modulation, the results lay a strong foundation for future research in neurotechnology and 
    interventions for ASD.

    Future Research - The current implementation of the BBCI framework can be extended to simulate more complex neural systems,
    incorporate more advanced machine learning algorithms, and explore the potential for neural modulation in various brain regions.
    Future work could focus on enhancing the model's accuracy and robustness by integrating additional neural data sources,
    such as EEG or fMRI, and employing deep learning techniques like convolutional neural networks (CNNs) or recurrent neural
    networks (RNNs). Furthermore, exploring the impact of different neural connectivity patterns and their role in cognitive
    functions could provide deeper insights. Additionally, ongoing research in neuroplasticity, neurodegenerative disorders,
    and cognitive enhancement can benefit from the insights gained through this study, particularly by applying the model to predict
    disease progression, develop personalized treatment plans, and design brain-computer interfaces for therapeutic interventions.
