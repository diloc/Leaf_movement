# Leaf_movement
Color distortion is an inherent problem in image-based phenotyping systems that are illuminated by artificial light. This distortion is problematic when examining plants because it can cause data to be incorrectly interpreted. One of the leading causes of color distortion is the non-uniform spectral and spatial distribution of artificial light. However, color correction algorithms currently used in plant phenotyping assume that a single and uniform illuminant causes color distortion. These algorithms are consequently inadequate to correct the local color distortion caused by multiple illuminants common in plant phenotyping systems, such as fluorescent tubes and LED light arrays. We describe here a color constancy algorithm, ColorBayes, based on Bayesian inference that corrects local color distortions. The algorithm estimates the local illuminants using the Bayes' rule, the maximum a posteriori, the observed image data, and prior illuminant information. The prior is obtained from light measurements and Macbeth ColorChecker charts located on the scene. 
