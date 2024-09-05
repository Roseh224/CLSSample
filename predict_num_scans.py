"""
### Overview: The SGM Beamline utilizes a machine that scans samples, to determine the sample's composition. The scans contain noise, but when 
you collect enough scans and combine them, then the noise is no longer an issue. The scans of some samples contain more noise than the 
scans of other samples. The samples whose scans contain more noise require more scans. This module (predict_num_scans) takes an initial batch 
of 10 sample scans. The module then uses another module (not written by me) to assign a numeric value to the amount of noise in a scan. This 
module (predict_num_scans) determines how much the noise levels vary in the first batch of 10 scans. It then uses this information to predict
how much noise future scans will contain. Based on these predictions, the module will tell the user how many more scans are necesary to 
reach a sufficiently low level of noise, and provide them with a graph indicating the predicted noise levels for this sample.###
"""


def check_sample_fitness(list_of_files):
    """
    Purpose: Create an SGMData object from hdf5 files. Interpolate the data in the SGMData object. Return this
    interpolated data.
    Parameters:
        list_of_files(list of str): the names of the hdf5 files to create an SGMData object out of.
    Returns:
        interp_list(list): a list of pandas dataframes. Contains the interpolated version of the data in the files
        specified in list_of_files.
    """
    sgm_data = sgmdata.load.SGMData(list_of_files)

    if len(sgm_data.__dict__['scans']) == 0:
        raise ValueError("hdf5 file must contain scans to be able to predict the number of scans required. The hdf5 "
                         "file you have provided does not contain any scans. PLease try again with an hdf5 file that"
                         " does contain scans.")
    has_sdd = False
    file = list(sgm_data.__dict__['scans'].keys())
    sample_name = list(sgm_data.__dict__['scans'][file[0]].__dict__.keys())
    signals = list(sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0]).__getattr__('signals').keys())
    i = 0
    while i < len(signals) and not has_sdd:
        if "sdd" in signals[i]:
            has_sdd = True
        else:
            i += 1
    if not has_sdd:
        raise ValueError("Scans must have sdd values to be able to predict the number of scans required. One or "
                         "more of the scans you have provided do not have sdd values. Please try again using "
                         "scans with sdd values. ")
    sample_type = sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']
    for indiv_file in file:
        sample_name = list(sgm_data.__dict__['scans'][indiv_file].__dict__.keys())
        for scan in sample_name:
            if sgm_data.__dict__['scans'][indiv_file].__getitem__(scan)['sample'] != sample_type:
                raise ValueError("In order to predict, the scans in the hdf5 file passed in by user must all be from"
                                 " the same sample. The scans in the hdf5 file passed in by the user are not all from"
                                 " the same sample. Please "
                                 "try again with an hdf5 file only containing scans from the"
                                 " same sample. ")
    interp_list = sgm_data.interpolate(resolution=0.1)
    return interp_list


def determine_num_scans(d_list, indices, desired_difference=0.17961943):
    """
    Purpose: takes a string of variations in noise levels and a desired noise level, and returns the number of additional
    variations (ie, scans) required to reach the desired noise level.
    Variables:
        d_list(list of floats): a list of the differences between our first ten acceptable scans. As our
        determine_scan_num function continues it will add values to the d_list.
        indices(list of ints): the scan numbers of the scans used to generate d_list.
        desired_noise(float): the variance between 5 consecutive scans the user would like to achieve, ie, we'll need to
        continue scans until this level of variance is reached. Default is the highest variance among sample scans.
    Returns:
         num_predictions+1(int): the number of scans required to reach the desired level of variance. Adding one
         because, since we're working differences in d_list, and a difference comes from 2 values, to get any number of
         differences you'll need that number of scans plus 1.
    """
    copied_indices = indices.copy()
    num_predictions = 9
    keep_predicting = True
    recent_differences = d_list[:5]
    variances = []
    recent_variances = []
    if len(d_list) < 9:
        raise ValueError("Prediction can only be made with hdf5 file containing 10 or more scans. Less than 10 scans"
                         " in the hdf5 file passed in. Please try again with an hdf5 file containing 10 or more scans.")
    # Checking if the desired level of variance has already been reached in the first 10 scans.
    for element in d_list:
        if len(recent_differences) < 10:
            recent_differences.append(element)
        elif len(recent_differences) == 10:
            recent_differences.append(element)
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                return 0
    # Predicting variances until we predict a variance that's at or below the desired level of variance.
    while keep_predicting:
        # Predicting the next difference.
        predicted_level = predict(d_list, copied_indices)
        copied_indices.append(int(copied_indices[-1]) + 1)
        num_predictions = num_predictions + 1
        # Adding the newly predicted differance to our list of variances.
        d_list = np.append(d_list, predicted_level[-1])
        # Calculating the variance of the newest differences.
        recent_variances.clear()
        recent_differences.pop(0)
        recent_differences.append(predicted_level[-1])
        for var in recent_differences:
            recent_variances.append(((var - np.mean(recent_differences)) ** 2))
        variances.append((np.sum(recent_variances)) / len(recent_variances))
        # Stopping the predictions if the desired level of variance has already been reached.
        if variances[-1] <= desired_difference:
            keep_predicting = False
        if num_predictions > 60:
            # If the desired_difference was the default value, not input by user.
            if desired_difference == 0.17961943:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            else:
                raise ValueError("Desired level of variance cannot be reached.")
    return num_predictions


def determine_num_scans(d_list, indices, desired_difference):
    """
    ### Description:
    -----
        Before the predict_num_scans function reaches this function, numeric values have already been assigned to the
        noise levels of the initial 10 sample scans. This function uses the noise levels of these 10 scans to predict 
        the noise levels of future scans. The predicted noise levels are then used to determine how many additional 
        scans are needed to have an average variance of desired_difference across the most recent 10 scans.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- At this point, the list contains the average variance between noise
        levels of the initial batch of scans provided by the user. During this function, the predicted levels of noise 
        for future scans is added to this list. 
        > **indices** *(type: list of ints)* -- The indexes of the initial batch of scans within d_list. So, if there 
        were 10 scans provided by the user initially, the list would contain the numbers 0-9.
        > **desired_noise** *(type: float)* -- the amnount of variance between 10 consecutive scans the user would like to
        achieve, ie, we'll need to continue scaning until this level of variance is reached.
    ### Returns:
    -----
        > **num_predictions + 1** *(type: int)*: The number of scans required to reach the user's desired level of variance, 
        plus 1. We add one, because we're working with the differences between 2 values.
    """
    # Make a copy of the indices list to avoid modifying the original
    copied_indices = indices.copy()
    # Initialize variables
    num_predictions = 9
    keep_predicting = True
    recent_differences = d_list[:5]
    variances = []
    recent_variances = []

    # Iterate through the elements in d_list
    for element in d_list:
        if len(recent_differences) < 10:
            recent_differences.append(element)
        elif len(recent_differences) == 10:
            recent_differences.append(element)
    
            # Calculate variances for the recent 10 differences        
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))

            # If desired_difference already reached within initial 10 scans, function indicates that no more scans are needed
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                return 0

    # Continue predicting until the desired level of variance is reached
    while keep_predicting:
        # Predict noise level for the next scan using predict() function
        predicted_level = predict(d_list, copied_indices)
        # Update indices and number of predictions
        copied_indices.append(int(copied_indices[-1]) + 1)
        num_predictions = num_predictions + 1
        # Update d_list with predicted noise level
        d_list = np.append(d_list, predicted_level[-1])
        # Clear recent_variances and update recent_differences for next set of calculations
        recent_variances.clear()
        recent_differences.pop(0)
        recent_differences.append(predicted_level[-1])
        
        # Calculate variances for most recent 10 differences
        for var in recent_differences:
            recent_variances.append(((var - np.mean(recent_differences)) ** 2))

        # Calculate average variance for most recent 10 differences
        variances.append((np.sum(recent_variances)) / len(recent_variances))
        
        # Check if the desired level of variance is reached
        if variances[-1] <= desired_difference:
            keep_predicting = False

        # Check for a maximum number of predictions to avoid infinite loops
        if num_predictions > 60:
            if desired_difference == 0.17961943:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            else:
                raise ValueError("Desired level of variance cannot be reached.")

    # Return the total number of predictions needed to reach the desired level of variance
    return num_predictions


def predict_num_scans(data, verbose=False, percent_of_log=0.4, num_scans=10):
    """
    ### Description:
    -----
        Takes the SGMData object (an object containing information on a sample including information on the noise levels 
        of the initial 10 scans of the sample) of a sample and uses a combination of other functions to predict how many 
        additional scans should be taken of that sample.
    ### Args:
    -----
        > **data** *(type: SGMData object)* -- The SGMData object for the sample the user would like the module to
            predict the number of scans required for. 
        > **verbose** *(type: optional boolean)* -- Default value is False. If set to True, gives user additional data
            on how the additional number of scans needed was calculated.
        > **percent_of_log** *(type: optional float)* -- Default value is 0.4. The level of noise that the user deems 
            sufficiently low for scanning to end. Ie,  The average of the noise values of the first ten scans is taken, 
            and the log of it is found. Scans continue to be taken, and the average of the noise values of the most 
            recent ten scans is taken. The log of this average is taken,and if it's less than percent_of_log multiplied
            by the log of the first ten averages, then scanning stops.
        > **num_scans** *(type: optional int)* -- Default value is 10. The number of scans from the scans provided by
            the user that the user would like to be used to predict the number of additional scans to take.
    ### Returns:
    -----
        >*(type: int)*: The predicted number of additional scans that should be taken of a sample.
    """
    # Use the check_sample_fitness function to make sure the data provided by the user is suitable.
    interp_list = check_sample_fitness(data)
    # Extract the necessary information from data provided by user
    file = list(data.__dict__['scans'].keys())
    sample_name = list(data.__dict__['scans'][file[0]].__dict__.keys())
    sample_type = data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']

    # Make sure the correct number of scans are being interpreted
    if num_scans >= (len(interp_list)):
        num_scans = len(interp_list)
    
    # Extract necessary data for prediction
    returned_data = extracting_data(interp_list[:num_scans])
    returned_diff_list_listed = [item for item in returned_data[0]]
    # Determine what amount variance between scans must be reached in order for sufficiently low noise levels
    cut_off_point_info = predict_cut_off(returned_diff_list_listed[: num_scans - 1], percent_of_log)
    # Determine how many scans must be taken in order to reach the sufficiently low noise level identified by predict_cut_off function
    number_of_scans = find_cut_off(returned_diff_list_listed[: num_scans - 1], cut_off_point_info[2])
    # Create a list of scan indices up to the total number of scans needed so that the predicted data can be plotted on a graph for the user
    i = 1
    num_scans_listed = []
    while i <= number_of_scans[0]:
        num_scans_listed.append(i)
        i += 1

    # If the user has requested additional information (by setting the "Verbose" to true) additional data is provided in text and graph form
    if verbose:
        print(
            " *** Messages starting with \" ***\" are messages containing additional data, other than the number of "
            "additional scans needed." + "\n *** Average of initial 10 values: " + str(cut_off_point_info[0]) +
            "\n *** Log of average of initial 10 values: " + str(cut_off_point_info[1]) +
            "\n *** Cut off val, based on log of average of initial 10 values: " + str(cut_off_point_info[2]) +
            "\n *** Cut-off at scan number: " + str(number_of_scans[0]) +
            "\n *** Value at scan " + str(number_of_scans[0]) + "(scans at which cut-off point is reached): " +
            str(number_of_scans[1][-1]))
        plot_predicted(num_scans_listed[num_scans - 1:], number_of_scans[1], cut_off_point_info[2], interp_list,
                       sample_type, num_scans)

    # Indicate the number of additional scans required for the sample
    return number_of_scans[0] - 10
