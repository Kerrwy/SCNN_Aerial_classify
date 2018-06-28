from importlib import import_module
import os


def Initial_spiking_model(filepath):

    if filepath is not None:
        assert os.path.isfile(filepath), \
            "Configuration file not found at {}.".format(filepath)
        config = update_setup(filepath)

        num_timesteps = int(config.get('simulation', 'duration'))
        batch_size = int(config.get('simulation', 'batch_size'))
        dataset_path = config.get('paths', 'dataset_path')

        from INI_temporal_mean_rate_target_sim import SNN
        spiking_model = SNN(config, None)

        return spiking_model, num_timesteps, batch_size, dataset_path
    return "config path not found"


def load_config(filepath):
    """
    Load a config file from ``filepath``.
    """

    try:
        import configparser
    except ImportError:
        # noinspection PyPep8Naming
        import ConfigParser as configparser
        # noinspection PyUnboundLocalVariable
        configparser = configparser

    assert os.path.isfile(filepath), \
        "Configuration file not found at {}.".format(filepath)

    config = configparser.ConfigParser()
    config.read(filepath)

    return config


def update_setup(config_filepath):
    """Update default settings with user settings and check they are valid.

    Load settings from configuration file at ``config_filepath``, and check that
    parameter choices are valid. Non-specified settings are filled in with
    defaults.
    """

    from textwrap import dedent

    # Load defaults.
    config = load_config(os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'config_defaults')))

    # Overwrite with user settings.
    config.read(config_filepath)

    keras_backend = config.get('simulation', 'keras_backend')
    keras_backends = eval(config.get('restrictions', 'keras_backends'))
    assert keras_backend in keras_backends, \
        "Keras backend {} not supported. Choose from {}.".format(keras_backend,
                                                                 keras_backends)
    os.environ['KERAS_BACKEND'] = keras_backend
    # The keras import has to happen after setting the backend environment
    # variable!
    import keras.backend as k
    assert k.backend() == keras_backend, \
        "Keras backend set to {} in snntoolbox config file, but has already " \
        "been set to {} by a previous keras import. Set backend " \
        "appropriately in the keras config file.".format(keras_backend,
                                                         k.backend())
    if keras_backend == 'tensorflow':
        # Limit GPU usage of tensorflow.
        tf_config = k.tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        k.tensorflow_backend.set_session(k.tf.Session(config=tf_config))


    # Check that simulator choice is valid.
    simulator = config.get('simulation', 'simulator')
    simulators = eval(config.get('restrictions', 'simulators'))
    assert simulator in simulators, \
        "Simulator '{}' not supported. Choose from {}".format(simulator,
                                                              simulators)

    # Warn user that it is not possible to use Brian2 simulator by loading a
    # pre-converted network from disk.
    if simulator == 'brian2' and not config.getboolean('tools', 'convert'):
        print(dedent("""\n
            SNN toolbox Warning: When using Brian 2 simulator, you need to
            convert the network each time you start a new session. (No
            saving/reloading methods implemented.) Setting convert = True.
            \n"""))
        config.set('tools', 'convert', str(True))

    # Set default path if user did not specify it.
    if config.get('paths', 'path_wd') == '':
        config.set('paths', 'path_wd', os.path.dirname(config_filepath))

    # Check specified working directory exists.
    path_wd = config.get('paths', 'path_wd')
    assert os.path.exists(path_wd), \
        "Working directory {} does not exist.".format(path_wd)

    # Check that choice of input model library is valid.
    model_lib = config.get('input', 'model_lib')
    model_libs = eval(config.get('restrictions', 'model_libs'))
    assert model_lib in model_libs, "ERROR: Input model library '{}' ".format(
        model_lib) + "not supported yet. Possible values: {}".format(model_libs)


    # Set default path if user did not specify it.
    if config.get('paths', 'dataset_path') == '':
        config.set('paths', 'dataset_path', os.path.dirname(__file__))

    # Check that the data set path is valid.
    dataset_path = os.path.abspath(config.get('paths', 'dataset_path'))
    config.set('paths', 'dataset_path', dataset_path)
    assert os.path.exists(dataset_path), "Path to data set does not exist: " \
                                         "{}".format(dataset_path)

    # Check that data set path contains the data in the specified format.
    assert os.listdir(dataset_path), "Data set directory is empty."
    normalize = config.getboolean('tools', 'normalize')
    dataset_format = config.get('input', 'dataset_format')

    if dataset_format == 'npz' and not (os.path.exists(os.path.join(
            dataset_path, 'x_test.npz')) and os.path.exists(os.path.join(
            dataset_path, 'y_test.npz'))):
        raise RuntimeWarning(
            "Data set file 'x_test.npz' or 'y_test.npz' was not found in "
            "specified data set path {}.".format(dataset_path))

    sample_idxs_to_test = eval(config.get('simulation', 'sample_idxs_to_test'))
    num_to_test = config.getint('simulation', 'num_to_test')
    if not sample_idxs_to_test == []:
        if len(sample_idxs_to_test) != num_to_test:
            print(dedent("""
            SNN toolbox warning: Settings mismatch. Adjusting 'num_to_test' to
            equal the number of 'sample_idxs_to_test'."""))
            config.set('simulation', 'num_to_test',
                       str(len(sample_idxs_to_test)))

    # Create log directory if it does not exist.
    if config.get('paths', 'log_dir_of_current_run') == '':
        config.set('paths', 'log_dir_of_current_run', os.path.join(
            path_wd, 'log', 'gui', config.get('paths', 'runlabel')))
    log_dir_of_current_run = config.get('paths', 'log_dir_of_current_run')
    if not os.path.isdir(log_dir_of_current_run):
        os.makedirs(log_dir_of_current_run)


    if simulator != 'INI' and not config.getboolean('input', 'poisson_input'):
        config.set('input', 'poisson_input', str(True))
        print(dedent("""\
            SNN toolbox Warning: Currently, turning off Poisson input is
            only possible in INI simulator. Falling back on Poisson input."""))

    # Make sure the number of samples to test is not lower than the batch size.
    batch_size = config.getint('simulation', 'batch_size')
    if config.getint('simulation', 'num_to_test') < batch_size:
        print(dedent("""\
            SNN toolbox Warning: 'num_to_test' set lower than 'batch_size'.
            In simulators that test samples batch-wise (e.g. INIsim), this
            can lead to undesired behavior. Setting 'num_to_test' equal to
            'batch_size'."""))
        config.set('simulation', 'num_to_test', str(batch_size))

    plot_var = get_plot_keys(config)
    plot_vars = eval(config.get('restrictions', 'plot_vars'))
    assert all([v in plot_vars for v in plot_var]), \
        "Plot variable(s) {} not understood.".format(
            [v for v in plot_var if v not in plot_vars])
    if 'all' in plot_var:
        plot_vars_all = plot_vars.copy()
        plot_vars_all.remove('all')
        config.set('output', 'plot_vars', str(plot_vars_all))

    log_var = get_log_keys(config)
    log_vars = eval(config.get('restrictions', 'log_vars'))
    assert all([v in log_vars for v in log_var]), \
        "Log variable(s) {} not understood.".format(
            [v for v in log_var if v not in log_vars])
    if 'all' in log_var:
        log_vars_all = log_vars.copy()
        log_vars_all.remove('all')
        config.set('output', 'log_vars', str(log_vars_all))

    # Change matplotlib plot properties, e.g. label font size
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
        if len(plot_vars) > 0:
            import warnings
            warnings.warn("Package 'matplotlib' not installed; disabling "
                          "plotting. Run 'pip install matplotlib' to enable "
                          "plotting.", ImportWarning)
            config.set('output', 'plot_vars', str({}))
    if matplotlib is not None:
        matplotlib.rcParams.update(eval(config.get('output', 'plotproperties')))

    # Check settings for parameter sweep
    param_name = config.get('parameter_sweep', 'param_name')
    try:
        config.get('cell', param_name)
    except KeyError:
        print("Unkown parameter name {} to sweep.".format(param_name))
        raise RuntimeError
    if not eval(config.get('parameter_sweep', 'param_values')):
        config.set('parameter_sweep', 'param_values',
                   str([eval(config.get('cell', param_name))]))

    spike_code = config.get('conversion', 'spike_code')
    spike_codes = eval(config.get('restrictions', 'spike_codes'))
    assert spike_code in spike_codes, \
        "Unknown spike code {} selected. Choose from {}.".format(spike_code,
                                                                 spike_codes)
    if spike_code == 'temporal_pattern':
        num_bits = str(config.getint('conversion', 'num_bits'))
        config.set('simulation', 'duration', num_bits)
        config.set('simulation', 'batch_size', '1')
    elif 'ttfs' in spike_code:
        config.set('cell', 'tau_refrac',
                   str(config.getint('simulation', 'duration')))
        config.set('conversion', 'softmax_to_relu', 'True')
    assert keras_backend != 'theano' or spike_code == 'temporal_mean_rate', \
        "Keras backend 'theano' only works when the 'spike_code' parameter " \
        "is set to 'temporal_mean_rate' in snntoolbox config."
    with open(os.path.join(log_dir_of_current_run, '.config'), str('w')) as f:
        config.write(f)

    return config


def get_log_keys(config):
    return set(eval(config.get('output', 'log_vars')))


def get_plot_keys(config):
    return set(eval(config.get('output', 'plot_vars')))


def get_abs_path(filepath, config):
    """Get an absolute path, possibly using current toolbox working dir.

    Parameters
    ----------

    filepath: str
        Filename or relative or absolute path. If only the filename is given,
        file is assumed to be in current working directory
        (``config.get('paths', 'path_wd')``). Non-absolute paths are interpreted
        relative to working dir.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    path: str
        Absolute path to file.

    """

    path, filename = os.path.split(filepath)
    if path == '':
        path = config.get('paths', 'path_wd')
    elif not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config.get('paths', 'path_wd'),
                                            path))
    return path, filename

def echo(text):
    """python 2 version of print(end='', flush=True)."""
    import sys

    sys.stdout.write(u'{}'.format(text))
    sys.stdout.flush()

def get_type(layer):
    """Get type of Keras layer.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Keras layer.

    Returns
    -------

    : str
        Layer type.

    """

    return layer.__class__.__name__

def set_time(scnn_model, t):
    """Set the simulation time variable of all layers in the network.

    Parameters
    ----------

    t: float
        Current simulation time.
    """
    import numpy as np
    for layer in scnn_model.snn.layers[1:]:
        if layer.get_time() is not None:  # Has time attribute
            layer.set_time(np.float32(t))

def reset(scnn_model, sample_idx):
    for layer in scnn_model.snn.layers[1:]:  # Skip input layer
        layer.reset(sample_idx)