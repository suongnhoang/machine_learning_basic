from .utils import np

def he_normal(weight_shape):
    """
    Initialize weights according `He normal` distribution. With mean = 0, std = sqrt(2 / num_input)
    """
    if len(weight_shape) == 4:
        fW, fH, fC, _ = weight_shape
        return np.random.normal(0, np.sqrt(2 / (fW*fH*fC)), weight_shape)
    num_input, _ = weight_shape
    return np.random.normal(0, np.sqrt(2 / num_input), weight_shape)

def he_uniform(weight_shape):
    """
    Initialize weights according `He uniform` distribution within the range [-limit, limit].
                With limit = sqrt(6 / num_input)
    """
    if len(weight_shape) == 4:
        fW, fH, fC, _ = weight_shape
        return np.random.uniform(-np.sqrt(6 / (fW*fH*fC)), np.sqrt(6 / (fW*fH*fC)), weight_shape)
    num_input, _ = weight_shape
    return np.random.uniform(-np.sqrt(6 / num_input), np.sqrt(6 / num_input), weight_shape)

def glorot_normal(weight_shape):
    """
    Initialize weights according `Xavier normal` distribution. With mean = 0, std = sqrt(2 / (num_input + num_output))
    """
    if len(weight_shape) == 4:
        fW, fH, fC, num_fitls = weight_shape
        return np.random.normal(0, np.sqrt(2 / (fW*fH*fC + num_fitls)), weight_shape)
    num_input, num_output = weight_shape
    return np.random.normal(0, np.sqrt(2 / (num_input + num_output)), weight_shape)

def glorot_uniform(weight_shape):
    """
    Initialize weights according `Xavier uniform` distribution within the range [-limit, limit].
                With limit = sqrt(6 / (num_input + num_output))
    """
    if len(weight_shape) == 4:
        fW, fH, fC, num_fitls = weight_shape
        return np.random.uniform(-np.sqrt(6 / (fW*fH*fC + num_fitls)), np.sqrt(6 / (fW*fH*fC + num_fitls)), weight_shape)
    num_input, num_output = weight_shape
    return np.random.uniform(-np.sqrt(6 / (num_input + num_output)), np.sqrt(6 / (num_input + num_output)), weight_shape)

def standard_normal(weight_shape):
    """
    Initialize weights according standard normal distribution with mean 0 variance 1.
    """
    return np.random.normal(size=weight_shape)


initialization_mapping = {"glorot_normal": glorot_normal, "he_normal": he_normal, 
                          "glorot_normal": glorot_normal, "std": standard_normal,
                          "he_uniform": he_uniform, "glorot_uniform": glorot_uniform}