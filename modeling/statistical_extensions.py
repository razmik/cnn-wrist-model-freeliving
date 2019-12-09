import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
from scipy.stats.stats import pearsonr


class EnergyTransform:

    @staticmethod
    def met_to_intensity(data):

        data = np.where(data < 1.5, 1, data)
        data = np.where((1.5 <= data) & (data < 3), 2, data)
        data = np.where(3 <= data, 3, data)

        return data


class BlandAltman:
    """
    Example method call:
    waist_ee = results.loc[(results['waist_ee'] >= 1) & (results['waist_ee'] < 3)]['waist_ee'].as_matrix()
    statistical_extensions.BlandAltman.plot_histogram(waist_ee, 300, 'Histogram of Energy Expenditure Value Distribution',
                                                  'Energy Expenditure (MET) from Waist', 9)
    """

    @staticmethod
    def get_min_regularised_data_per_subject(data):

        min_count = min(data.groupby(['subject'])['waist_ee'].count())
        return data.groupby('subject').head(min_count)

    @staticmethod
    def annotate_bland_altman_paired_plot(dataframe):

        for label, x, y in zip(dataframe['subject'], dataframe['mean'], dataframe['diff']):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    @staticmethod
    def get_antilog(log_val):
        return round(math.pow(10, log_val), 2)

    @staticmethod
    def plot_graph(plot_number, plot_title, x_values, y_values, upper_loa, mean_bias, lower_loa, output_filename):

        x_label = 'Reference Energy Expenditure (METs)'
        y_label = 'Difference (Prediction - Reference) (log METs)'

        cat = output_filename.split('_')[-1]
        if cat == 'sb':
            x_lim = (1.0, 1.75)
            y_lim = (-1.2, 1.2)
        elif cat == 'lpa':
            x_lim = (1, 3.5)
            y_lim = (-1.2, 1.2)
        elif cat == 'sblpa':
            x_lim = (1, 3.5)
            y_lim = (-1.2, 1.2)
        elif cat == 'mvpa':
            x_lim = (2.5, 15)
            y_lim = (-1.2, 1.2)
        elif cat == 'mpa':
            x_lim = (2.5, 6.5)
            y_lim = (-1.2, 1.2)
        elif cat == 'vpa':
            x_lim = (5.5, 15)
            y_lim = (-1.2, 1.2)

        x_annotate_begin = 10.4
        y_gap = 0.05
        ratio_suffix = ''

        plt.figure(dpi=1200)
        # plt.title(plot_title)
        plt.scatter(x_values, y_values)

        # Black: #000000
        plt.axhline(upper_loa, color='gray', linestyle='dotted')
        plt.axhline(mean_bias, color='gray', linestyle='--')
        plt.axhline(lower_loa, color='gray', linestyle='dotted')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # http://www.anjuke.tech/questions/843083/matplotlib-savefig-in-jpeg-format

        plt.savefig(output_filename + '.png', dpi=1200)

        plt.annotate(str(BlandAltman.get_antilog(upper_loa)) + ratio_suffix, xy=(x_annotate_begin, (upper_loa + y_gap)))
        plt.annotate(str(BlandAltman.get_antilog(mean_bias)) + ratio_suffix, xy=(x_annotate_begin, (mean_bias + y_gap)))
        plt.annotate(str(BlandAltman.get_antilog(lower_loa)) + ratio_suffix, xy=(x_annotate_begin, (lower_loa + y_gap)))

        # print(output_filename)
        assessment_result = '\n\n' + output_filename
        assessment_result += '\nupper_loa:' + str(BlandAltman.get_antilog(upper_loa))
        assessment_result += '\nmean_bias:' + str(BlandAltman.get_antilog(mean_bias))
        assessment_result += '\nlower_loa:' + str(BlandAltman.get_antilog(lower_loa))

        Utils.print_assessment_results(output_filename + '_ba_stats.txt', assessment_result)

        plt.savefig(output_filename + '_annotated.png', dpi=1200)
        plt.clf()
        plt.close()

    @staticmethod
    def bland_altman_paired_plot_tested(dataframe, plot_title, plot_number, log_transformed=False,
                                        min_count_regularise=False, output_filename=''):

        """Define multiple dataframes based on the activity intensity"""
        dataframe_sb = dataframe.loc[dataframe['waist_ee'] <= 1.5]
        dataframe_lpa = dataframe.loc[(1.5 < dataframe['waist_ee']) & (dataframe['waist_ee'] < 3)]
        dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]
        dataframe_mpa = dataframe.loc[(3 <= dataframe['waist_ee']) & (dataframe['waist_ee'] < 6)]
        dataframe_vpa = dataframe.loc[6 <= dataframe['waist_ee']]

        """
        Process BA plot for Overall
        """
        _, mean_bias_sb_overall, upper_loa_sb_overall, lower_loa_sb_overall = BlandAltman._bland_altman_analyse(
            dataframe, log_transformed=log_transformed, min_count_regularise=min_count_regularise)

        assessment_result = '\n\n' + output_filename
        assessment_result += '\nupper_loa:' + str(BlandAltman.get_antilog(upper_loa_sb_overall))
        assessment_result += '\nmean_bias:' + str(BlandAltman.get_antilog(mean_bias_sb_overall))
        assessment_result += '\nlower_loa:' + str(BlandAltman.get_antilog(lower_loa_sb_overall))
        Utils.print_assessment_results(output_filename + '_overall_ba_stats.txt', assessment_result)

        """
        Process BA plot for SB
        """
        dataframe_sb, mean_bias_sb, upper_loa_sb, lower_loa_sb = BlandAltman._bland_altman_analyse(
            dataframe_sb, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 1, plot_title + ' - SB - Williams Work-Energy (98)',
                               dataframe_sb['mean'], dataframe_sb['diff'],
                               upper_loa_sb, mean_bias_sb, lower_loa_sb,
                               output_filename + '_sb')

        """
        Process BA plot for LPA
        """
        dataframe_lpa, mean_bias_lpa, upper_loa_lpa, lower_loa_lpa = BlandAltman._bland_altman_analyse(
            dataframe_lpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 3, plot_title + ' - LPA - Williams Work-Energy (98)',
                               dataframe_lpa['mean'], dataframe_lpa['diff'],
                               upper_loa_lpa, mean_bias_lpa, lower_loa_lpa,
                               output_filename + '_lpa')

        """
        Process BA plot for MVPA
        """
        dataframe_mvpa, mean_bias_mvpa, upper_loa_mvpa, lower_loa_mvpa = BlandAltman._bland_altman_analyse(
            dataframe_mvpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 4, plot_title + ' - MVPA - Freedson VM3 Combination (11)',
                               dataframe_mvpa['mean'], dataframe_mvpa['diff'],
                               upper_loa_mvpa, mean_bias_mvpa, lower_loa_mvpa,
                               output_filename + '_mvpa')

        """
        Process BA plot for MPA
        """
        dataframe_mpa, mean_bias_mpa, upper_loa_mpa, lower_loa_mpa = BlandAltman._bland_altman_analyse(
            dataframe_mpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 5, plot_title + ' - MPA - Freedson VM3 Combination (11)',
                               dataframe_mpa['mean'], dataframe_mpa['diff'],
                               upper_loa_mpa, mean_bias_mpa, lower_loa_mpa,
                               output_filename + '_mpa')

        """
        Process BA plot for VPA
        """
        dataframe_vpa, mean_bias_vpa, upper_loa_vpa, lower_loa_vpa = BlandAltman._bland_altman_analyse(
            dataframe_vpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 6, plot_title + ' - VPA - Freedson VM3 Combination (11)',
                               dataframe_vpa['mean'], dataframe_vpa['diff'],
                               upper_loa_vpa, mean_bias_vpa, lower_loa_vpa,
                               output_filename + '_vpa')

    @staticmethod
    def bland_altman_paired_plot_for_two_catagories(dataframe, plot_title, plot_number, log_transformed=False,
                                                    min_count_regularise=False, output_filename=''):

        """Define multiple dataframes based on the activity intensity"""
        dataframe_sb_lpa = dataframe.loc[dataframe['waist_ee'] < 3]
        dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]
        dataframe_mpa = dataframe.loc[(3 <= dataframe['waist_ee']) & (dataframe['waist_ee'] < 6)]
        dataframe_vpa = dataframe.loc[6 <= dataframe['waist_ee']]

        """
        Process BA plot for Overall
        """
        _, mean_bias_sb_overall, upper_loa_sb_overall, lower_loa_sb_overall = BlandAltman._bland_altman_analyse(
            dataframe, log_transformed=log_transformed, min_count_regularise=min_count_regularise)

        assessment_result = '\n\n' + output_filename
        assessment_result += '\nupper_loa:' + str(BlandAltman.get_antilog(upper_loa_sb_overall))
        assessment_result += '\nmean_bias:' + str(BlandAltman.get_antilog(mean_bias_sb_overall))
        assessment_result += '\nlower_loa:' + str(BlandAltman.get_antilog(lower_loa_sb_overall))
        Utils.print_assessment_results(output_filename + '_overall_ba_stats.txt', assessment_result)

        """
        Process BA plot for SB + LPA
        """
        dataframe_sb_lpa, mean_bias_sb_lpa, upper_loa_sb_lpa, lower_loa_sb_lpa = BlandAltman._bland_altman_analyse(
            dataframe_sb_lpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 1, plot_title + ' - SB+LPA - Williams Work-Energy (98)',
                               dataframe_sb_lpa['mean'], dataframe_sb_lpa['diff'],
                               upper_loa_sb_lpa, mean_bias_sb_lpa, lower_loa_sb_lpa,
                               output_filename + '_sblpa')

        """
       Process BA plot for MVPA
       """
        dataframe_mvpa, mean_bias_mvpa, upper_loa_mvpa, lower_loa_mvpa = BlandAltman._bland_altman_analyse(
            dataframe_mvpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 4, plot_title + ' - MVPA - Freedson VM3 Combination (11)',
                               dataframe_mvpa['mean'], dataframe_mvpa['diff'],
                               upper_loa_mvpa, mean_bias_mvpa, lower_loa_mvpa,
                               output_filename + '_mvpa')

        """
        Process BA plot for MPA
        """
        dataframe_mpa, mean_bias_mpa, upper_loa_mpa, lower_loa_mpa = BlandAltman._bland_altman_analyse(
            dataframe_mpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 5, plot_title + ' - MPA - Freedson VM3 Combination (11)',
                               dataframe_mpa['mean'], dataframe_mpa['diff'],
                               upper_loa_mpa, mean_bias_mpa, lower_loa_mpa,
                               output_filename + '_mpa')

        """
        Process BA plot for VPA
        """
        dataframe_vpa, mean_bias_vpa, upper_loa_vpa, lower_loa_vpa = BlandAltman._bland_altman_analyse(
            dataframe_vpa, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        BlandAltman.plot_graph(plot_number + 6, plot_title + ' - VPA - Freedson VM3 Combination (11)',
                               dataframe_vpa['mean'], dataframe_vpa['diff'],
                               upper_loa_vpa, mean_bias_vpa, lower_loa_vpa,
                               output_filename + '_vpa')

    @staticmethod
    def _bland_altman_analyse(dataframe, log_transformed=False, min_count_regularise=False):

        if min_count_regularise:
            dataframe = BlandAltman.get_min_regularised_data_per_subject(dataframe)

        if log_transformed:
            dataframe = dataframe.assign(waist_ee_log_transformed=np.log10(dataframe['waist_ee_cleaned']))
            dataframe = dataframe.assign(predicted_ee_log_transformed=np.log10(dataframe['predicted_ee_cleaned']))

        # x-axis: use the reference/freedson vm3
        dataframe = dataframe.assign(mean=dataframe.as_matrix(columns=['waist_ee_cleaned']))

        # x-axis: use the mean value of reference/predicted
        # dataframe = dataframe.assign(mean=np.mean([dataframe.as_matrix(columns=['waist_ee_cleaned']),
        #                                            dataframe.as_matrix(columns=['predicted_ee_cleaned'])], axis=0))
        dataframe = dataframe.assign(
            diff=dataframe['predicted_ee_log_transformed'] - dataframe['waist_ee_log_transformed'])
        # dataframe = dataframe.assign(diff=dataframe['waist_ee_cleaned']/dataframe['predicted_ee_cleaned'])

        k = len(pd.unique(dataframe.subject))  # number of conditions
        N = len(dataframe.values)  # conditions times participants

        DFbetween = k - 1
        DFwithin = N - k
        DFtotal = N - 1

        anova_data = pd.DataFrame()
        dataframe_summary = dataframe.groupby(['subject'])
        anova_data['count'] = dataframe_summary['diff'].count()  # number of values in each group ng
        anova_data['sum'] = dataframe_summary['diff'].sum()  # sum of values in each group
        anova_data['mean'] = dataframe_summary['diff'].mean()  # mean of values in each group Xg
        anova_data['variance'] = dataframe_summary['diff'].var()
        anova_data['sd'] = np.sqrt(anova_data['variance'])
        anova_data['count_sqr'] = anova_data['count'] ** 2

        grand_mean = anova_data['sum'].sum() / anova_data['count'].sum()  # XG

        # Calculate the MSS within
        squared_within = 0
        for name, group in dataframe_summary:
            group_mean = group['diff'].sum() / group['diff'].count()

            squared = 0
            for index, row in group.iterrows():
                squared += (row['diff'] - group_mean) ** 2

            squared_within += squared

        SSwithin = squared_within

        # Calculate the MSS between
        ss_between_partial = 0
        for index, row in anova_data.iterrows():
            ss_between_partial += row['count'] * ((row['mean'] - grand_mean) ** 2)

        SSbetween = ss_between_partial

        #  Calculate SS total
        squared_total = 0
        for index, row in dataframe.iterrows():
            squared_total += (row['diff'] - grand_mean) ** 2

        SStotal = squared_total

        MSbetween = SSbetween / DFbetween
        MSwithin = SSwithin / DFwithin

        n = DFbetween + 1
        m = DFtotal + 1
        sigma_m2 = sum(anova_data['count_sqr'])

        variance_b_method = MSwithin

        diff_bet_within = MSbetween - MSwithin
        divisor = (m ** 2 - sigma_m2) / ((n - 1) * m)
        variance = diff_bet_within / divisor

        total_variance = variance + variance_b_method
        sd = np.sqrt(total_variance)

        mean_bias = sum(anova_data['sum']) / m
        upper_loa = mean_bias + (1.96 * sd)
        lower_loa = mean_bias - (1.96 * sd)

        return dataframe, mean_bias, upper_loa, lower_loa

    @staticmethod
    def clean_data_points(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])

        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1

        return data

    @staticmethod
    def clean_data_points_reference_only(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_transformed=data['waist_ee'])

        return data

    @staticmethod
    def clean_data_points_prediction_only(data, prediction_column):

        data.loc[(data[prediction_column] < 0), prediction_column] = 0

        return data


class GeneralStats:
    """
    Confidence Interval:
        https://github.com/cmrivers/epipy/blob/master/epipy/analyses.py
        https://www.medcalc.org/calc/diagnostic_test.php
        https://www.wikihow.com/Calculate-95%25-Confidence-Interval-for-a-Test%27s-Sensitivity
    """

    @staticmethod
    def evaluation_statistics(confusion_matrix):

        if confusion_matrix.shape == (3, 3):

            # Name the values
            tpa = confusion_matrix[0, 0]
            tpb = confusion_matrix[1, 1]
            tpc = confusion_matrix[2, 2]
            eab = confusion_matrix[0, 1]
            eac = confusion_matrix[0, 2]
            eba = confusion_matrix[1, 0]
            ebc = confusion_matrix[1, 2]
            eca = confusion_matrix[2, 0]
            ecb = confusion_matrix[2, 1]

            # Calculate accuracy for label 1
            total_classifications = sum(sum(confusion_matrix))
            accuracy = (tpa + tpb + tpc) / total_classifications
            accuracy_se = np.sqrt((accuracy * (1 - accuracy)) / total_classifications)
            accuracy_confidence_interval = (accuracy - (1.96 * accuracy_se), accuracy + (1.96 * accuracy_se))

            # Calculate Precision for label 1
            precisionA = tpa / (tpa + eba + eca)

            # Calculate Sensitivity for label 1
            sensitivityA = tpa / (tpa + eab + eac)
            senA_se = np.sqrt((sensitivityA * (1 - sensitivityA)) / (tpa + eab + eac))
            sensitivityA_confidence_interval = (sensitivityA - (1.96 * senA_se), sensitivityA + (1.96 * senA_se))

            # Calculate Specificity for label 1
            tna = tpb + ebc + ecb + tpc
            specificityA = tna / (tna + eba + eca)
            specA_se = np.sqrt((specificityA * (1 - specificityA)) / (tna + eba + eca))
            specificityA_confidence_interval = (specificityA - (1.96 * specA_se), specificityA + (1.96 * specA_se))

            # Calculate Precision for label 2
            precisionB = tpb / (tpb + eab + ecb)

            # Calculate Sensitivity for label 2
            sensitivityB = tpb / (tpb + eba + ebc)
            senB_se = np.sqrt((sensitivityB * (1 - sensitivityB)) / (tpb + eba + ebc))
            sensitivityB_confidence_interval = (sensitivityB - (1.96 * senB_se), sensitivityB + (1.96 * senB_se))

            # Calculate Specificity for label 2
            tnb = tpa + eac + eca + tpc
            specificityB = tnb / (tnb + eab + ecb)
            specB_se = np.sqrt((specificityB * (1 - specificityB)) / (tnb + eab + ecb))
            specificityB_confidence_interval = (specificityB - (1.96 * specB_se), specificityB + (1.96 * specB_se))

            # Calculate Precision for label 2
            precisionC = tpc / (tpc + eac + ebc)

            # Calculate Sensitivity for label 2
            sensitivityC = tpc / (tpc + eca + ecb)
            senC_se = np.sqrt((sensitivityC * (1 - sensitivityC)) / (tpc + eca + ecb))
            sensitivityC_confidence_interval = (sensitivityC - (1.96 * senC_se), sensitivityC + (1.96 * senC_se))

            # Calculate Specificity for label 2
            tnc = tpa + eab + eba + tpb
            specificityC = tnc / (tnc + eac + ebc)
            specC_se = np.sqrt((specificityC * (1 - specificityC)) / (tnc + eac + ebc))
            specificityC_confidence_interval = (specificityC - (1.96 * specC_se), specificityC + (1.96 * specC_se))

            round_digits = 4

            sensitivityA_confidence_interval = (round(sensitivityA_confidence_interval[0], round_digits),
                                                round(sensitivityA_confidence_interval[1], round_digits))
            sensitivityB_confidence_interval = (round(sensitivityB_confidence_interval[0], round_digits),
                                                round(sensitivityB_confidence_interval[1], round_digits))
            sensitivityC_confidence_interval = (round(sensitivityC_confidence_interval[0], round_digits),
                                                round(sensitivityC_confidence_interval[1], round_digits))
            specificityA_confidence_interval = (round(specificityA_confidence_interval[0], round_digits),
                                                round(specificityA_confidence_interval[1], round_digits))
            specificityB_confidence_interval = (round(specificityB_confidence_interval[0], round_digits),
                                                round(specificityB_confidence_interval[1], round_digits))
            specificityC_confidence_interval = (round(specificityC_confidence_interval[0], round_digits),
                                                round(specificityC_confidence_interval[1], round_digits))

            return {
                'accuracy': round(accuracy, round_digits),
                'accuracy_ci': (round(accuracy_confidence_interval[0], round_digits),
                                round(accuracy_confidence_interval[1], round_digits)),
                'precision': [round(precisionA, round_digits), round(precisionB, round_digits),
                              round(precisionC, round_digits)],
                'recall': [round(sensitivityA, round_digits), round(sensitivityB, round_digits),
                           round(sensitivityC, round_digits)],
                'sensitivity': [round(sensitivityA, round_digits), round(sensitivityB, round_digits),
                                round(sensitivityC, round_digits)],
                'specificity': [round(specificityA, round_digits), round(specificityB, round_digits),
                                round(specificityC, round_digits)],
                'sensitivity_ci': [sensitivityA_confidence_interval, sensitivityB_confidence_interval,
                                   sensitivityC_confidence_interval],
                'specificity_ci': [specificityA_confidence_interval, specificityB_confidence_interval,
                                   specificityC_confidence_interval]
            }

        elif confusion_matrix.shape == (2, 2):

            tpa = confusion_matrix[0, 0]
            tpb = confusion_matrix[1, 1]
            eab = confusion_matrix[0, 1]
            eba = confusion_matrix[1, 0]

            # Calculate accuracy for label 1
            total_classifications = sum(sum(confusion_matrix))
            accuracy = (tpa + tpb) / total_classifications
            accuracy_se = np.sqrt((accuracy * (1 - accuracy)) / total_classifications)
            accuracy_confidence_interval = (accuracy - (1.96 * accuracy_se), accuracy + (1.96 * accuracy_se))

            # Calculate Precision for label 1
            precisionA = tpa / (tpa + eba)

            # Calculate Sensitivity for label 1
            sensitivityA = tpa / (tpa + eab)
            senA_se = np.sqrt((sensitivityA * (1 - sensitivityA)) / (tpa + eab))
            sensitivityA_confidence_interval = (sensitivityA - (1.96 * senA_se), sensitivityA + (1.96 * senA_se))

            # Calculate Specificity for label 1
            tna = tpb
            specificityA = tna / (tna + eba)
            specA_se = np.sqrt((specificityA * (1 - specificityA)) / (tna + eba))
            specificityA_confidence_interval = (specificityA - (1.96 * specA_se), specificityA + (1.96 * specA_se))

            # Calculate Precision for label 2
            precisionB = tpb / (tpb + eab)

            # Calculate Sensitivity for label 2
            sensitivityB = tpb / (tpb + eba)
            senB_se = np.sqrt((sensitivityB * (1 - sensitivityB)) / (tpb + eba))
            sensitivityB_confidence_interval = (sensitivityB - (1.96 * senB_se), sensitivityB + (1.96 * senB_se))

            # Calculate Specificity for label 2
            tnb = tpa
            specificityB = tnb / (tnb + eab)
            specB_se = np.sqrt((specificityB * (1 - specificityB)) / (tnb + eab))
            specificityB_confidence_interval = (specificityB - (1.96 * specB_se), specificityB + (1.96 * specB_se))

            round_digits = 4

            sensitivityA_confidence_interval = (round(sensitivityA_confidence_interval[0], round_digits),
                                                round(sensitivityA_confidence_interval[1], round_digits))
            sensitivityB_confidence_interval = (round(sensitivityB_confidence_interval[0], round_digits),
                                                round(sensitivityB_confidence_interval[1], round_digits))
            specificityA_confidence_interval = (round(specificityA_confidence_interval[0], round_digits),
                                                round(specificityA_confidence_interval[1], round_digits))
            specificityB_confidence_interval = (round(specificityB_confidence_interval[0], round_digits),
                                                round(specificityB_confidence_interval[1], round_digits))

            return {
                'accuracy': round(accuracy, round_digits),
                'accuracy_ci': (round(accuracy_confidence_interval[0], round_digits),
                                round(accuracy_confidence_interval[1], round_digits)),
                'precision': [round(precisionA, round_digits), round(precisionB, round_digits)],
                'recall': [round(sensitivityA, round_digits), round(sensitivityB, round_digits)],
                'sensitivity': [round(sensitivityA, round_digits), round(sensitivityB, round_digits)],
                'specificity': [round(specificityA, round_digits), round(specificityB, round_digits)],
                'sensitivity_ci': [sensitivityA_confidence_interval, sensitivityB_confidence_interval],
                'specificity_ci': [specificityA_confidence_interval, specificityB_confidence_interval]
            }

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues, output_filename=''):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title + " Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print(title + ' confusion matrix')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(output_filename, dpi=1200)
        plt.clf()
        plt.close()

    """
     Parameters
     ----------
     x : 1D array
     y : 1D array the same length as x
    
     Returns
     -------
     (Pearson's correlation coefficient, 2-tailed p-value)
    """

    @staticmethod
    def pearson_correlation(x, y):
        return pearsonr(x, y)


class Average_Stats:

    @staticmethod
    def include_general_userdata(data, user_data):
        data = pd.merge(data, user_data, on='subject', how='outer')
        data['bmi_catagory'] = 'none'
        data.loc[data['bmi'] < 18.5, 'bmi_catagory'] = 'underweight'
        data.loc[(data['bmi'] >= 18.5) & (data['bmi'] < 25), 'bmi_catagory'] = 'normal'
        data.loc[(data['bmi'] >= 25) & (data['bmi'] < 30), 'bmi_catagory'] = 'overweight'
        data.loc[data['bmi'] >= 30, 'bmi_catagory'] = 'obese'
        return data

    @staticmethod
    def evaluate_average_measures(data, epoch, output_title, output_folder_path, general_user_details=None):

        def get_averaged_df(dataset, count_field, new_col, multiplyer):
            dataset_count = dataset.groupby(['subject'])[count_field].count().reset_index(name=new_col)
            dataset_count[new_col] *= (multiplyer / (60 * 60))
            return dataset_count

        def get_average_counted_df(data_actual, data_predict, mul):
            return pd.merge(get_averaged_df(data_actual, 'waist_ee', 'actual_time', mul),
                            get_averaged_df(data_predict, 'predicted_ee', 'predicted_time', mul),
                            on='subject', how='outer')

        round_digits = 2
        mul = int(epoch.split('Epoch')[1])

        # Evaluate SB
        df_sb = get_average_counted_df(data.loc[data['waist_ee'] <= 1.5], data.loc[data['predicted_ee'] <= 1.5], mul)
        sb_actual_avg = str(round(df_sb['actual_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['actual_time'].std(), round_digits))
        sb_predicted_avg = str(round(df_sb['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['predicted_time'].std(), round_digits))

        # Evaluate LPA
        df_lpa = get_average_counted_df(data.loc[(data['waist_ee'] > 1.5) & (data['waist_ee'] < 3)],
                                        data.loc[(data['predicted_ee'] > 1.5) & (data['predicted_ee'] < 3)], mul)
        lpa_actual_avg = str(round(df_lpa['actual_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['actual_time'].std(), round_digits))
        lpa_predicted_avg = str(round(df_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['predicted_time'].std(), round_digits))

        # Evaluate SB+LPA
        df_sb_lpa = get_average_counted_df(data.loc[(data['waist_ee'] < 3)], data.loc[(data['predicted_ee'] < 3)], mul)
        sb_lpa_actual_avg = str(df_sb_lpa['actual_time'].mean()) + "+-" + str(df_sb_lpa['actual_time'].std())
        sb_lpa_predicted_avg = str(round(df_sb_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb_lpa['predicted_time'].std(), round_digits))

        # Evaluate MVPA
        df_mvpa = get_average_counted_df(data.loc[data['waist_ee'] >= 3], data.loc[data['predicted_ee'] >= 3], mul)
        mvpa_actual_avg = str(df_mvpa['actual_time'].mean()) + "+-" + str(df_mvpa['actual_time'].std())
        mvpa_predicted_avg = str(round(df_mvpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_mvpa['predicted_time'].std(), round_digits))

        if general_user_details is not None:
            df_sb = Average_Stats.include_general_userdata(df_sb, general_user_details)
            df_sb.to_csv(output_folder_path + output_title + '_sb_averaged.csv', index=False)
            df_lpa = Average_Stats.include_general_userdata(df_lpa, general_user_details)
            df_lpa.to_csv(output_folder_path + output_title + '_lpa_averaged.csv', index=False)
            df_sb_lpa = Average_Stats.include_general_userdata(df_sb_lpa, general_user_details)
            df_sb_lpa.to_csv(output_folder_path + output_title + '_sb_lpa_averaged.csv', index=False)
            df_mvpa = Average_Stats.include_general_userdata(df_mvpa, general_user_details)
            df_mvpa.to_csv(output_folder_path + output_title + '_mvpa_averaged.csv', index=False)

        return [sb_actual_avg, sb_predicted_avg], [lpa_actual_avg, lpa_predicted_avg], [sb_lpa_actual_avg,
                                                                                        sb_lpa_predicted_avg], [
                   mvpa_actual_avg, mvpa_predicted_avg]

    @staticmethod
    def evaluate_average_measures_for_categorical(data, epoch, output_title, output_folder_path, general_user_details):
        def get_averaged_df(dataset, count_field, new_col, multiplyer):
            dataset_count = dataset.groupby(['subject'])[count_field].count().reset_index(name=new_col)
            dataset_count[new_col] *= (multiplyer / (60 * 60))
            return dataset_count

        def get_average_counted_df(data_actual, data_predict, mul):
            return pd.merge(get_averaged_df(data_actual, 'waist_ee', 'actual_time', mul),
                            get_averaged_df(data_predict, 'predicted_category', 'predicted_time', mul),
                            on='subject', how='outer')

        round_digits = 2
        mul = int(epoch.split('Epoch')[1])

        # Evaluate SB
        df_sb = get_average_counted_df(data.loc[data['waist_ee'] <= 1.5], data.loc[data['predicted_category'] == 1],
                                       mul)
        sb_actual_avg = str(df_sb['actual_time'].mean()) + "+-" + str(df_sb['actual_time'].std())
        sb_predicted_avg = str(round(df_sb['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['predicted_time'].std(), round_digits))

        # Evaluate LPA
        df_lpa = get_average_counted_df(data.loc[(data['waist_ee'] > 1.5) & (data['waist_ee'] < 3)],
                                        data.loc[data['predicted_category'] == 2], mul)
        lpa_actual_avg = str(df_lpa['actual_time'].mean()) + "+-" + str(df_lpa['actual_time'].std())
        lpa_predicted_avg = str(round(df_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['predicted_time'].std(), round_digits))

        # Evaluate SB+LPA
        df_sb_lpa = get_average_counted_df(data.loc[(data['waist_ee'] < 3)], data.loc[(data['predicted_category'] != 3)], mul)
        sb_lpa_actual_avg = str(df_sb_lpa['actual_time'].mean()) + "+-" + str(df_sb_lpa['actual_time'].std())
        sb_lpa_predicted_avg = str(round(df_sb_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb_lpa['predicted_time'].std(), round_digits))

        # Evaluate MVPA
        df_mvpa = get_average_counted_df(data.loc[data['waist_ee'] >= 3], data.loc[data['predicted_category'] == 3],
                                         mul)
        mvpa_actual_avg = str(df_mvpa['actual_time'].mean()) + "+-" + str(df_mvpa['actual_time'].std())
        mvpa_predicted_avg = str(round(df_mvpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_mvpa['predicted_time'].std(), round_digits))

        if general_user_details is not None:
            df_sb = Average_Stats.include_general_userdata(df_sb, general_user_details)
            df_sb.to_csv(output_folder_path + output_title + '_sb_averaged.csv', index=False)
            df_lpa = Average_Stats.include_general_userdata(df_lpa, general_user_details)
            df_lpa.to_csv(output_folder_path + output_title + '_lpa_averaged.csv', index=False)
            df_sb_lpa = Average_Stats.include_general_userdata(df_sb_lpa, general_user_details)
            df_sb_lpa.to_csv(output_folder_path + output_title + '_sb_lpa_averaged.csv', index=False)
            df_mvpa = Average_Stats.include_general_userdata(df_mvpa, general_user_details)
            df_mvpa.to_csv(output_folder_path + output_title + '_mvpa_averaged.csv', index=False)

        return [sb_actual_avg, sb_predicted_avg], [lpa_actual_avg, lpa_predicted_avg], [sb_lpa_actual_avg,
                                                                                        sb_lpa_predicted_avg], [
                   mvpa_actual_avg, mvpa_predicted_avg]

    @staticmethod
    def evaluate_average_measures_controller(data, epoch, output_title, output_folder_path, is_categorical=False):

        def evaluate(category, filtered_data, epoch, assessment_result, output_title=None, general_user_details=None):

            if not is_categorical:
                sb, lpa, sb_lpa, mvpa = Average_Stats.evaluate_average_measures(filtered_data, epoch, output_title, output_folder_path, general_user_details)
            else:
                sb, lpa, sb_lpa, mvpa = Average_Stats.evaluate_average_measures_for_categorical(filtered_data, epoch, output_title, output_folder_path, general_user_details)

            assessment_result += '\n\n' + category + ' Assessment of Average time\n\n'
            assessment_result += 'SB actual:\t' + sb[0] + '\n'
            assessment_result += 'SB predicted:\t' + sb[1] + '\n'
            assessment_result += 'LPA actual:\t' + lpa[0] + '\n'
            assessment_result += 'LPA predicted:\t' + lpa[1] + '\n'
            assessment_result += 'SB+LPA actual:\t' + sb_lpa[0] + '\n'
            assessment_result += 'SB+LPA predicted:\t' + sb_lpa[1] + '\n'
            assessment_result += 'MVPA actual:\t' + mvpa[0] + '\n'
            assessment_result += 'MVPA predicted:\t' + mvpa[1] + '\n'

            return assessment_result

        results_output_str = ''
        general_user_details = pd.read_csv('E:/Projects/accelerometer-project/analyze/user_details.csv')
        data = pd.merge(data, general_user_details, on='subject', how='outer')

        # Overall evaluation
        results_output_str = evaluate('Overall', data, epoch, results_output_str, output_title, general_user_details)

        # Do the evaluation for Sex
        male_data = data.loc[data['gender'] == 'Male']
        results_output_str = evaluate('Male', male_data, epoch, results_output_str)

        female_data = data.loc[data['gender'] == 'Female']
        results_output_str = evaluate('Female', female_data, epoch, results_output_str)

        female_data = data.loc[data['gender'] == 'Other']
        results_output_str = evaluate('Other', female_data, epoch, results_output_str)

        # Do the evaluation based on BMI catagory
        underweight_data = data.loc[data['bmi'] < 18.5]
        results_output_str = evaluate('BMI - underweight', underweight_data, epoch, results_output_str)

        normal_data = data.loc[(data['bmi'] >= 18.5) & (data['bmi'] < 25)]
        results_output_str = evaluate('BMI - normal weight', normal_data, epoch, results_output_str)

        overweight_data = data.loc[(data['bmi'] >= 25) & (data['bmi'] < 30)]
        results_output_str = evaluate('BMI - overweight', overweight_data, epoch, results_output_str)

        obesity_data = data.loc[data['bmi'] >= 30]
        results_output_str = evaluate('BMI - obesity', obesity_data, epoch, results_output_str)

        results_output_filename = output_folder_path + output_title + '_average_time_assessment.txt'
        Utils.print_assessment_results(results_output_filename, results_output_str)


class Utils:

    @staticmethod
    def print_assessment_results(output_filename, result_string):
        with open(output_filename, "w") as text_file:
            text_file.write(result_string)


class ReferenceMethod:

    @staticmethod
    def update_reference_ee(dataframe):

        def get_freedson_vm3_combination_11_energy_expenditure(row):

            if dataframe['waist_vm_cpm'][row.name] < 2453:
                # METs = 0.001092(VA) + 1.336129  [capped at 2.9999, where VM3 < 2453]
                met_value = (0.001092 * dataframe['waist_cpm'][row.name]) + 1.336129
                met_value = met_value if met_value < 2.9999 else 2.9999
            else:
                # METs = 0.000863(VM3) + 0.668876 [where VM3 ≥ 2453]
                met_value = 0.000863 * dataframe['waist_vm_60'][row.name] + 0.668876

            return met_value

        # Update reference waist ee
        dataframe['waist_ee'] = dataframe.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)

        return dataframe
