import os
import numpy as np
import bilby
import argparse
import scipy
from helper_functions import (
    load_folder,
    whiten_bandpass_bkgs,
    olib_time_domain_sine_gaussian,
    inject_hplus_hcross,
    get_background_segs,
    WNB,
    clean_gw_events,
    get_loud_segments,
    slice_bkg_segments
)

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEG_NUM_TIMESTEPS,
    SAMPLE_RATE,
    STRAIN_START,
    N_TRAIN_INJECTIONS,
    N_FM_INJECTIONS,
    DATA_SEGMENT_LOAD_START,
    DATA_SEGMENT_LOAD_STOP,
    TRAIN_INJECTION_SEGMENT_LENGTH,
    FM_INJECTION_SEGMENT_LENGTH,
    BBH_WINDOW_LEFT,
    BBH_WINDOW_RIGHT,
    BBH_AMPLITUDE_BAR,
    BBH_N_SAMPLES,
    SG_WINDOW_LEFT,
    SG_WINDOW_RIGHT,
    SG_AMPLITUDE_BAR,
    SG_N_SAMPLES,
    GLITCH_WINDOW_LEFT,
    GLITCH_WINDOW_RIGHT,
    GLITCH_N_SAMPLES,
    GLITCH_AMPLITUDE_BAR,
    BKG_N_SAMPLES,
    FM_INJECTION_SNR,
    N_VARYING_SNR_INJECTIONS,
    VARYING_SNR_DISTRIBUTION,
    VARYING_SNR_LOW,
    VARYING_SNR_HIGH,
    VARYING_SNR_SEGMENT_INJECTION_LENGTH,
    CURRICULUM_SNRS,
    SNR_SN_LOW,
    SNR_SN_HIGH,
    LOADED_DATA_SAMPLE_RATE  # goes for the SN signals as well
)


def generate_timeslides(
        folder_path: str,
        event_times_path: str):
    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    event_times = np.load(event_times_path)

    data = data[:, np.newaxis, :]
    whitened = whiten_bandpass_bkgs(data, SAMPLE_RATE, loaded_data['H1'][
                                    'asd'], loaded_data['L1']['asd'])
    whitened = np.swapaxes(whitened, 0, 1)[0]  # batch dimension removed

    data_cleaned = clean_gw_events(event_times,
                                   whitened,
                                   STRAIN_START + DATA_SEGMENT_LOAD_START,
                                   STRAIN_START + DATA_SEGMENT_LOAD_STOP)
    return data_cleaned


def bbh_polarization_generator(
        n_injections,
        segment_length=2):

    bbh_waveform_args = dict(waveform_approximant='IMRPhenomPv2',
                             reference_frequency=50., minimum_frequency=20.)
    waveform = bilby.gw.source.lal_binary_black_hole
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=segment_length,
        sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=waveform,
        waveform_arguments=bbh_waveform_args
    )
    priors = bilby.gw.prior.BBHPriorDict()
    injection_parameters = priors.sample(n_injections)
    injection_parameters = [
        dict(zip(injection_parameters, col))
        for col in zip(*injection_parameters.values())
    ]
    for i, p in enumerate(injection_parameters):
        dist = np.random.uniform(50, 200)
        p['luminosity_distance'] = dist

    crosses = []
    plusses = []
    for i, p in enumerate(injection_parameters):
        p['geocent_time'] = 0
        pols = waveform_generator.time_domain_strain(p)
        cross = pols['cross']
        plus = pols['plus']

        crosses.append(cross)
        plusses.append(plus)

    crosses = np.vstack(crosses)
    plusses = np.vstack(plusses)

    half = crosses.shape[1] // 2

    crosses = np.hstack([crosses[:, half:], crosses[:, :half]])
    plusses = np.hstack([plusses[:, half:], plusses[:, :half]])

    return [crosses, plusses]


def sg_polarization_generator(
        n_injections,
        segment_length=2,
        prior_file='data/SG.prior'):

    waveform = olib_time_domain_sine_gaussian
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=segment_length,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=waveform,
        waveform_arguments=None
    )
    priors = bilby.gw.prior.PriorDict(prior_file)
    injection_parameters = priors.sample(n_injections)
    injection_parameters = [
        dict(zip(injection_parameters, col))
        for col in zip(*injection_parameters.values())
    ]
    crosses = []
    plusses = []
    for i, p in enumerate(injection_parameters):
        p['geocent_time'] = 0
        pols = waveform_generator.time_domain_strain(p)

        peak = np.argmax((pols['cross']))

        half = len(pols['cross']) // 2
        cross = pols['cross']
        plus = pols['plus']
        cross = np.concatenate(
            [cross[(half + peak):], cross[:(half + peak)]])
        plus = np.concatenate([plus[(half + peak):], plus[:(half + peak)]])
        crosses.append(cross)
        plusses.append(plus)

    crosses = np.vstack(crosses)
    plusses = np.vstack(plusses)

    return [crosses, plusses]


def wnb_polarization_generator(
        n_injections,
        fmin=400,
        fmax=1000,
        duration=0.1):  # in seconds

    # merge with background
    wnb_hplus = np.zeros((n_injections, SAMPLE_RATE))
    wnb_hcross = np.zeros((n_injections, SAMPLE_RATE))
    for i in range(n_injections):
        wnb = WNB(duration, SAMPLE_RATE, fmin, fmax,
                  enveloped=True, sidePad=(SAMPLE_RATE - SAMPLE_RATE * duration) // 2)
        if wnb[0].shape[0] == 4095:
            gen_wnb_hplus = np.append(wnb[0], np.array([0]))
            gen_wnb_hcross = np.append(wnb[1], np.array([0]))
        elif wnb[0].shape[0] == 4094:
            gen_wnb_hplus = np.append(wnb[0], np.array([0, 0]))
            gen_wnb_hcross = np.append(wnb[1], np.array([0, 0]))
        else:
            gen_wnb_hplus = wnb[0]
            gen_wnb_hcross = wnb[1]

        wnb_hplus[i, :] = gen_wnb_hplus
        wnb_hcross[i, :] = gen_wnb_hcross

    return wnb_hcross, wnb_hplus


def inject_signal(
        folder_path: str,  # source of detector data, includes detector data and the omicron glitches/corresponding SNRs
        # source of the polarization files to be injected into the data
        data=None,
        SNR=None,
        # length of background segment to fetch for each injection
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH,
        inject_at_end=False,
        return_injection_snr=False):

    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack(
        [loaded_data['H1']['data'], loaded_data['L1']['data']])

    polarizations = []
    crosses, plusses = data
    for i in range(len(plusses)):
        polarizations.append({'plus': plusses[i], 'cross': crosses[i]})
    print(f'requested shape {len(polarizations) * 1}')
    bkg_segs = get_background_segs(loaded_data, detector_data, len(
        polarizations) * 1, segment_length)
    detector_psds = inject_hplus_hcross(bkg_segs[:, 0, :],
                                        polarizations[0],
                                        SAMPLE_RATE,
                                        segment_length,
                                        SNR=1,
                                        background=loaded_data,
                                        get_psds=True,
                                        inject_at_end=inject_at_end)
    print(f'background segments shape {bkg_segs.shape}')
    final_data = []
    sampled_SNR = []
    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        sample_snr = None
        if SNR is not None:
            sample_snr = SNR()
            sampled_SNR.append(sample_snr)
        for j in range(1):
            injected_waveform, _ = inject_hplus_hcross(bkg_segs[:, i, :],
                                                       pols,
                                                       SAMPLE_RATE,
                                                       segment_length,
                                                       SNR=sample_snr,
                                                       background=loaded_data,
                                                       detector_psds=detector_psds,
                                                       inject_at_end=inject_at_end)
            bandpass_segs = whiten_bandpass_bkgs(injected_waveform, SAMPLE_RATE, loaded_data[
                                                 'H1']['asd'], loaded_data['L1']['asd'])
            final_data.append(bandpass_segs)

    if return_injection_snr:
        return np.hstack(final_data), np.array(sampled_SNR)
    return np.hstack(final_data)


def inject_signal_curriculum(
        folder_path: str,  # source of detector data, includes detector data and the omicron glitches/corresponding SNRs
        # source of the polarization files to be injected into the data
        data=None,
        SNR=None,
        # length of background segment to fetch for each injection
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH,
        inject_at_end=False,
        return_injection_snr=False):

    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack(
        [loaded_data['H1']['data'], loaded_data['L1']['data']])

    polarizations = []
    crosses, plusses = data
    for i in range(len(plusses)):
        polarizations.append({'plus': plusses[i], 'cross': crosses[i]})

    bkg_segs = get_background_segs(loaded_data, detector_data, len(
        polarizations) * 1, segment_length)
    detector_psds = inject_hplus_hcross(bkg_segs[:, 0, :],
                                        polarizations[0],
                                        SAMPLE_RATE,
                                        segment_length,
                                        SNR=1,
                                        background=loaded_data,
                                        get_psds=True,
                                        inject_at_end=inject_at_end)

    sample_SNRS = np.zeros((len(polarizations), len(CURRICULUM_SNRS)))
    for j in range(len(CURRICULUM_SNRS)):
        center = CURRICULUM_SNRS[j]
        sample_SNRS[:, j] = np.random.uniform(
            center - center / 4, center + center / 2, len(polarizations))

    final_data_noise = []
    final_data_clean = []
    for i in range(len(CURRICULUM_SNRS)):
        final_data_noise.append([])
        final_data_clean.append([])

    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        injected_waveforms, clean_waveforms = inject_hplus_hcross(bkg_segs[:, i, :],
                                                                  pols,
                                                                  SAMPLE_RATE,
                                                                  segment_length,
                                                                  SNR=sample_SNRS[
                                                                      i],
                                                                  background=loaded_data,
                                                                  detector_psds=detector_psds,
                                                                  inject_at_end=inject_at_end)

        for j, injected_waveform in enumerate(injected_waveforms):
            noisy_bandpassed_segs = whiten_bandpass_bkgs(
                injected_waveform, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
            final_data_noise[j].append(noisy_bandpassed_segs)
        for j, clean_waveform in enumerate(clean_waveforms):
            clean_bandpassed_segs = whiten_bandpass_bkgs(
                clean_waveform, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
            final_data_clean[j].append(clean_bandpassed_segs)

    final_data_noise = np.array([np.hstack(elem) for elem in final_data_noise])
    final_data_clean = np.array([np.hstack(elem) for elem in final_data_clean])
    return final_data_noise, final_data_clean


def generate_backgrounds(
        folder_path: str,
        n_backgrounds: int,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH):

    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack(
        [loaded_data['H1']['data'], loaded_data['L1']['data']])

    bkg_segs = get_background_segs(
        loaded_data, detector_data, n_backgrounds, segment_length)
    whitened_segs = whiten_bandpass_bkgs(bkg_segs, SAMPLE_RATE, loaded_data[
                                         'H1']['asd'], loaded_data['L1']['asd'])
    return whitened_segs


def generate_glitches(
        folder_path: str,
        n_glitches: int,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH,
        load_start=DATA_SEGMENT_LOAD_START,
        load_stop=DATA_SEGMENT_LOAD_STOP):

    loaded_data = load_folder(
        folder_path,
        load_start,
        load_stop,
        glitches=True)

    if 'H1' not in loaded_data.keys() or 'L1' not in loaded_data.keys():
        return 'empty'

    detector_data = np.vstack(
        [loaded_data['H1']['data'], loaded_data['L1']['data']])

    N = n_glitches
    loud_times_H1 = get_loud_segments(loaded_data['H1'], N, segment_length)
    loud_times_L1 = get_loud_segments(loaded_data['L1'], N, segment_length)
    loud_times = np.union1d(loud_times_H1, loud_times_L1)
    glitch_segs, _ = slice_bkg_segments(loaded_data['H1'], detector_data, loud_times,
                                        segment_length)
    whitened_segs = whiten_bandpass_bkgs(glitch_segs, SAMPLE_RATE, loaded_data[
                                         'H1']['asd'], loaded_data['L1']['asd'])
    return whitened_segs


def sampler(
        data,
        n_samples: int,
        bound_low: int,
        bound_high: int,
        amplitude_bar: int,
        sample_len=SEG_NUM_TIMESTEPS):

    fill = np.empty((len(data) * n_samples, 2, sample_len))
    print('SAMPLER fill shape', fill.shape)
    print('len(data), n_samples, sample_len', len(data), n_samples, sample_len)
    midp = data.shape[-1] // 2
    filled_count = 0
    for n in range(len(data)):
        for j in range(n_samples):
            max_amp = -1
            attempts = 0
            while max_amp < amplitude_bar and attempts < n_samples * 2:
                attempts += 1
                start_index = int(np.random.uniform(
                    bound_low, bound_high - sample_len))
                start_index += midp
                segment = data[n, :, start_index:start_index + sample_len]
                max_amp = np.amax(np.abs(segment))
            if max_amp >= amplitude_bar:
                # at this point, it's passed the amplitude test
                fill[filled_count, :, :] = segment
                filled_count += 1

    return fill[:filled_count]


def sample_injections_main(
        source: str,  # directory containing the injection files
        # list of classes on which you would like to perform preparation for
        # training
        target_class: str,
        sample_len: int=SEG_NUM_TIMESTEPS,
        data=None):

    sampler_args = {
        'bbh': [BBH_N_SAMPLES, int(BBH_WINDOW_LEFT * SAMPLE_RATE),
                int(BBH_WINDOW_RIGHT * SAMPLE_RATE), BBH_AMPLITUDE_BAR],
        'sg': [SG_N_SAMPLES, int(SG_WINDOW_LEFT * SAMPLE_RATE),
               int(SG_WINDOW_RIGHT * SAMPLE_RATE), SG_AMPLITUDE_BAR],
        'background': [BKG_N_SAMPLES, None, None, 0],
        'glitch': [GLITCH_N_SAMPLES, int(GLITCH_WINDOW_LEFT * SAMPLE_RATE),
                   int(GLITCH_WINDOW_RIGHT * SAMPLE_RATE), GLITCH_AMPLITUDE_BAR]
    }

    data = data.swapaxes(0, 1)
    N_samples, bound_low, bound_high, amplitude_bar = sampler_args[
        target_class]

    if bound_low is None:
        assert bound_high is None
        midp = data.shape[-1] // 2
        bound_low = -midp
        bound_high = midp - sample_len

    training_data = sampler(
        data, N_samples, bound_low, bound_high, amplitude_bar)

    print('out from sample injections', training_data.shape)
    return training_data


def curriculum_sampler(
        data_noisy,
        data_clean,
        target_class: str,
        sample_len: int=SEG_NUM_TIMESTEPS):

    sampler_args = {
        'bbh': [BBH_N_SAMPLES, int(BBH_WINDOW_LEFT * SAMPLE_RATE),
                int(BBH_WINDOW_RIGHT * SAMPLE_RATE), BBH_AMPLITUDE_BAR],
        'sg': [SG_N_SAMPLES, int(SG_WINDOW_LEFT * SAMPLE_RATE),
               int(SG_WINDOW_RIGHT * SAMPLE_RATE), SG_AMPLITUDE_BAR]
    }
    N_samples_each, bound_low, bound_high, _ = sampler_args[target_class]

    SNR_ind, N_ifos, N_injections, inj_seg_len = data_noisy.shape
    midp = inj_seg_len // 2

    fill_noisy = np.zeros((SNR_ind, N_injections * N_samples_each,
                           N_ifos, sample_len))
    fill_clean = np.zeros(fill_noisy.shape)
    for s in range(SNR_ind):
        for n in range(N_injections):
            for m in range(N_samples_each):

                start = np.random.uniform(midp + bound_low,
                                          midp + bound_high - sample_len)
                start = int(start)
                fill_noisy[s, n * N_samples_each + m, :,
                           :] = data_noisy[s, :, n, start:start + sample_len]
                fill_clean[s, n * N_samples_each + m, :,
                           :] = data_clean[s, :, n, start:start + sample_len]

    return fill_noisy, fill_clean


def make_snr_sampler(distribution, low, hi):
    if distribution == 'uniform':
        def sampler():
            return np.random.randint(low, hi)
    else:
        print('Invalid or unimplemented distribution choice', distribution)
        assert False

    return sampler


def fetch_sn_polarization(path):
    # load the .txt files for all the phi, theta values as an array of cross and plus numpy arras
    # resample from the original 16384 down to 4096

    def reduce_name(name):
        # specific for powell
        mass, eos, phi, theta, _, mode = name.split('_')
        phi, theta, mode = phi[3:], theta[5:], mode[:-4]
        return mass, eos, phi, theta, mode

    cross = dict()
    plus = dict()

    for file in os.listdir(path):
        mass, eos, phi, theta, mode = reduce_name(file)
        iden = f'{phi}_{theta}'
        if mode == 'hcross':
            cross[iden] = np.loadtxt(f'{path}/{file}')
        else:
            assert mode == 'hplus'
            plus[iden] = np.loadtxt(f'{path}/{file}')
            load_len = len(plus[iden])  # for powell, they are all the same

    assert LOADED_DATA_SAMPLE_RATE == 16384
    downsample_len = int(load_len * SAMPLE_RATE / LOADED_DATA_SAMPLE_RATE)
    n_samples = len(plus.keys())
    assert len(cross.keys()) == n_samples

    cross_arr = np.zeros((n_samples, downsample_len))
    plus_arr = np.zeros((n_samples, downsample_len))
    for i, iden in enumerate(cross.keys()):
        assert iden in plus.keys()

        # downsample
        cross_pol = scipy.signal.resample(cross[iden], downsample_len)
        plus_pol = scipy.signal.resample(plus[iden], downsample_len)

        # add to arrays
        cross_arr[i] = cross_pol
        plus_arr[i] = plus_pol

    return cross_arr, plus_arr


def repeat_arr(arr, n):
    # repeat a 2-d array n times along axis 0
    fill = np.zeros((arr.shape[0] * n, arr.shape[1]))
    for i in range(n):
        fill[i * arr.shape[0]:(i + 1) * arr.shape[0], :] = arr

    return fill


def main(args):
    sampled_snr = None

    if args.stype == 'bbh':
        # 1: generate the polarization files for the signal classes of interest
        bbh_cross, bbh_plus = bbh_polarization_generator(N_TRAIN_INJECTIONS)

        # 2: create injections with those signal classes
        noisy, clean = inject_signal_curriculum(folder_path=args.folder_path,
                                                data=[bbh_cross, bbh_plus]
                                                )
        # 3: Turn the injections into segments, ready for training
        noisy_samples, clean_samples = curriculum_sampler(noisy, clean, 'bbh')
        training_data = dict(
            noisy=noisy_samples,
            clean=clean_samples)

    elif args.stype == 'sglf' or args.stype == 'sghf':
        # 1: generate the polarization files for the signal classes of interest
        sg_cross, sg_plus = sg_polarization_generator(N_TRAIN_INJECTIONS,
                                                      prior_file=f'data/{args.stype}.prior')

        # 2: create injections with those signal classes
        noisy, clean = inject_signal_curriculum(folder_path=args.folder_path,
                                                data=[sg_cross, sg_plus]
                                                )
        # 3: Turn the injections into segments, ready for training
        noisy_samples, clean_samples = curriculum_sampler(noisy, clean, 'sg')

        training_data = dict(
            noisy=noisy_samples,
            clean=clean_samples)

    elif args.stype == 'background':

        # 2.5: generate/fetch the background classes
        backgrounds = generate_backgrounds(folder_path=args.folder_path,
                                           n_backgrounds=N_TRAIN_INJECTIONS)
        # 3: Turn the injections into segments, ready for training
        training_data = sample_injections_main(source=None,
                                               target_class=args.stype,
                                               data=backgrounds)
        training_data = dict(data=training_data)

    elif args.stype == 'glitch':
        if args.start is not None:
            assert args.stop is not None
            glitches = generate_glitches(folder_path=args.folder_path,
                                         n_glitches=N_TRAIN_INJECTIONS,
                                         load_start=int(args.start), load_stop=int(args.stop))
        else:
            glitches = generate_glitches(folder_path=args.folder_path,
                                         n_glitches=N_TRAIN_INJECTIONS)

        if glitches == 'empty':
            return 'empty'

        training_data = sample_injections_main(source=None,
                                               target_class=args.stype,
                                               data=glitches)

        return training_data

    elif args.stype == 'glitches':
        segments = np.load(args.intersections)
        datums = []

        for segment in segments:
            start, stop = segment[0], segment[1]
            seglen = stop - start
            if seglen < 3600:
                continue

            full_path = f'./output/omicron/{start}_{stop}/'

            for j in range(seglen // 3600):
                split_start, split_stop = j * 3600, (j + 1) * 3600
                j_args = argparse.Namespace(
                    folder_path=full_path,
                    save_file=f'{full_path}/glitch.npy',
                    stype='glitch',
                    start=split_start,
                    stop=split_stop)

                datum = main(j_args)
                if datum == 'empty':
                    continue

                datums.append(datum)

        training_data = np.concatenate(datums, axis=0)
        print(f'Total glitch shape is {training_data.shape}')
        training_data = dict(data=training_data)

    elif args.stype == 'timeslides':
        event_times_path = 'data/LIGO_EVENT_TIMES.npy'
        training_data = generate_timeslides(
            args.folder_path, event_times_path=event_times_path)
        training_data = dict(data=training_data)

    elif args.stype == 'bbh_varying_snr' or args.stype == 'bbh_fm_optimization':
        # 1: generate the polarization files for the signal classes of interest
        bbh_cross, bbh_plus = bbh_polarization_generator(
            N_VARYING_SNR_INJECTIONS)

        sampler = make_snr_sampler(
            VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)
        # 2: create the injections with those signal classes
        BBH_injections, sampled_snr = inject_signal(folder_path=args.folder_path,
                                                    data=[bbh_cross, bbh_plus],
                                                    segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                                    inject_at_end=True,
                                                    SNR=sampler,
                                                    return_injection_snr=True)
        training_data = BBH_injections.swapaxes(0, 1)
        training_data = dict(data=training_data)

    elif args.stype == 'sglf_varying_snr' or args.stype == 'sghf_varying_snr' or args.stype == 'sglf_fm_optimization' or args.stype == 'sghf_fm_optimization':
        # 1: generate the polarization files for the signal classes of interest
        sg_cross, sg_plus = sg_polarization_generator(N_VARYING_SNR_INJECTIONS,
                                                      prior_file=f'data/{args.stype[:4]}.prior')

        sampler = make_snr_sampler(
            VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)
        # 2: create the injections with those signal classes
        sg_injections, sampled_snr = inject_signal(folder_path=args.folder_path,
                                                   data=[sg_cross, sg_plus],
                                                   segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                                   inject_at_end=True,
                                                   SNR=sampler,
                                                   return_injection_snr=True)
        training_data = sg_injections.swapaxes(0, 1)
        training_data = dict(data=training_data)

    elif args.stype == 'wnblf_varying_snr' or args.stype == 'wnbhf_varying_snr' or args.stype == 'wnblf_fm_optimization' or args.stype == 'wnbhf_fm_optimization':

        if 'wnblf' in args.stype:
            fmin = 40
            fmax = 400
        else:
            fmin = 400
            fmax = 1000

        # 1: generate the polarization files for the signal classes of interest
        wnb_cross, wnb_plus = wnb_polarization_generator(
            N_VARYING_SNR_INJECTIONS,
            fmin=fmin,
            fmax=fmax)

        sampler = make_snr_sampler(
            VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)

        # 2: create the injections with those signal classes
        training_data, sampled_snr = inject_signal(folder_path=args.folder_path,
                                                   data=[wnb_cross, wnb_plus],
                                                   segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                                   inject_at_end=True,
                                                   SNR=sampler,
                                                   return_injection_snr=True)
        training_data = dict(data=training_data)

    elif args.stype == 'supernova_varying_snr' or args.stype == 'supernova_fm_optimization':
        # 1 : Fetch the polarization files
        sn_cross, sn_plus = fetch_sn_polarization(args.sn_polarization_path)

        # copy the array to get more samples, approximately equal to N_VARYING_SNR_INJECTIONS
        #'uniform' prior over the cross and plus, so just copy each one some number of times
        n_repeat = int(N_VARYING_SNR_INJECTIONS / len(sn_cross))
        sn_cross, sn_plus = repeat_arr(
            sn_cross, n_repeat), repeat_arr(sn_plus, n_repeat)

        sampler = make_snr_sampler(
            VARYING_SNR_DISTRIBUTION, SNR_SN_LOW, SNR_SN_HIGH)

        # 2: create injections with those polarizations
        training_data, sampled_snr = inject_signal(folder_path=args.folder_path,
                                                   data=[sn_cross, sn_plus],
                                                   segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                                   inject_at_end=True,
                                                   SNR=sampler,
                                                   return_injection_snr=True)
        training_data = dict(data=training_data)

    elif args.stype == 'wnbhf' or args.stype == 'wnblf':

        if 'wnblf' in args.stype:
            fmin = 40
            fmax = 400
        else:
            fmin = 400
            fmax = 1000

        wnb_cross, wnb_plus = wnb_polarization_generator(10,
                                                         fmin=fmin,
                                                         fmax=fmax)
        # 2: create injections with those signal classes
        noisy, clean = inject_signal_curriculum(
            folder_path=args.folder_path,
            data=[wnb_cross, wnb_plus])

        training_data = dict(data=clean)

    elif args.stype == 'supernova':
         # 1 : Fetch the polarization files
        sn_cross, sn_plus = fetch_sn_polarization(args.sn_polarization_path)

        noisy, clean = inject_signal_curriculum(
            folder_path=args.folder_path,
            data=[sn_cross, sn_plus])

        training_data = dict(data=clean)

    np.savez(args.save_file, **training_data)

    if sampled_snr is not None:
        # drop it in between name and .npy
        snr_save_path = f'{args.save_file[:-4]}_SNR{args.save_file[-4:]}'
        np.save(snr_save_path, sampled_snr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('folder_path', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('save_file', help='Where to save the file with injections',
                        type=str)

    parser.add_argument('--stype', help='Which type of the injection to generate',
                        type=str, choices=['bbh', 'sglf', 'sghf', 'background',
                                           'glitch', 'glitches', 'timeslides',
                                           'bbh_fm_optimization',
                                           'sghf_fm_optimization', 'sglf_fm_optimization',
                                           'bbh_varying_snr',
                                           'sghf_varying_snr', 'sglf_varying_snr',
                                           'wnblf', 'wnbhf',
                                           'wnbhf_varying_snr', 'wnblf_varying_snr',
                                           'wnblf_fm_optimization', 'wnbhf_fm_optimization',
                                           'supernova',
                                           'supernova_varying_snr', 'supernova_fm_optimization'])

    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--stop', type=str, default=None)
    parser.add_argument('--intersections', type=str, default=None)
    parser.add_argument('--sn-polarization-path', type=str,
                        default='data/z85_sfho/')
    args = parser.parse_args()
    main(args)
