''' Plotting and analysis functions.'''

import pylab as plt
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import ttest_rel, sem
from scipy.signal import find_peaks_cwt
from . import utility as ut

#----------------------------------------------------------------------------------
# Various analyses.
#----------------------------------------------------------------------------------

def session_plot(session, show_TO = False, ylabel = True):
    'Plot of choice moving average and reward block structure for single session.'
    choices, transitions, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, dtype = bool)
    second_steps = second_steps * 1.1-0.05
    mov_ave = ut.exp_mov_ave(choices)

    plt.plot(mov_ave,'k.-', markersize = 3)

    if hasattr(session, 'blocks'):
        #transitions = transitions == session.blocks['trial_trans_state'] # Convert transitions AB to transtions CR.
        for i in range(len(session.blocks['start_trials'])):
            y = [0.1,0.5,0.9][session.blocks['reward_states'][i]]  # y position coresponding to reward state.
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            if session.blocks['transition_states'][i]:
                plt.plot(x, [y,y], 'orange', linewidth = 2)
            else:
                y = 1 - y  # Invert y position if transition is inverted.
                plt.plot(x, [y,y], 'purple', linewidth = 2)    
    if show_TO:
        def symplot(y,guard,symbol):
            x_ind = np.where(guard)[0]
            plt.plot(x_ind,y[x_ind],symbol, markersize = 5)
        symplot(second_steps,  transitions &  outcomes,'ob' )
        symplot(second_steps,  transitions & ~outcomes,'xb')
        symplot(second_steps, ~transitions &  outcomes,'og')
        symplot(second_steps, ~transitions & ~outcomes,'xg')  
    plt.plot([0,len(choices)],[0.75,0.75],'--k')
    plt.plot([0,len(choices)],[0.25,0.25],'--k')

    plt.xlabel('Trial Number')
    plt.yticks([0,0.5,1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(0,len(choices))
    if ylabel:plt.ylabel('Choice moving average')

def runlength_analysis(sessions):
    'Histogram of length of runs of single choice.'
    run_lengths = []
    for session in sessions:
        choices = session.CTSO['choices']
        prev_choice = choices[0]
        run_length = 0
        for choice in choices[1:]:
            run_length += 1
            if not choice == prev_choice:
                run_lengths.append(run_length)
                run_length = 0
            prev_choice= choice
    counts,bins = np.histogram(run_lengths,list(range(1,41)))
    plt.plot(bins[:-1], counts/len(run_lengths))
    plt.ylabel('Fraction')
    plt.xlabel('Run length')
    print(('Mean run length: {}'.format(np.mean(run_lengths))))

#----------------------------------------------------------------------------------
# Longditudinal analyses through training.
#----------------------------------------------------------------------------------

def cumulative_trials(experiment, col = 'b'):
    'Cumulative trials as a function of session number with cross-animal SEM errorbars.'
    trials = np.zeros([experiment.n_subjects, experiment.n_days])
    blocks = np.zeros([experiment.n_subjects, experiment.n_days])
    for i, sID in enumerate(experiment.subject_IDs):
        for day in range(experiment.n_days):
            trials[i, day] =     experiment.get_sessions(sID, day + 1).n_trials
            blocks[i, day] = len(experiment.get_sessions(sID, day + 1).blocks['start_trials']) - 1
    cum_trials = np.cumsum(trials,1)
    mean_trials = np.mean(cum_trials,0)
    SEM_trials = sem(cum_trials,0)
    cum_blocks = np.cumsum(blocks,1)
    mean_blocks = np.mean(cum_blocks,0)
    SEM_blocks = sem(cum_blocks,0)
    days = np.arange(experiment.n_days) + 1
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(days, mean_trials)
    plt.fill_between(days, mean_trials - SEM_trials, mean_trials + SEM_trials, alpha = 0.2, facecolor = col)
    plt.subplot(2,1,2)
    plt.plot(days, mean_blocks)
    plt.fill_between(days, mean_blocks - SEM_blocks, mean_blocks + SEM_blocks, alpha = 0.2, facecolor = col)

def trials_per_block(experiment, use_blocks = 'non_neutral', clf = True, fig_no = 1, last_n_days = 6):
    ' Number of trials taken to finish each block.'
    days = set([s.day for s in experiment.get_sessions('all', 'all')])   
    mean_tpb, sd_tpb = ([],[])
    residual_trials = np.zeros(len(experiment.subject_IDs)) # Number of trials in last (uncompleted) block of session.
    days_trials_per_block = []
    for day in days:
        day_trials_per_block = []
        sessions = experiment.get_sessions('all', day)
        for session in sessions:
            assert hasattr(session,'blocks'), 'Session do not have block info.'
            ax = experiment.subject_IDs.index(session.subject_ID) # Animal index used for residual trials array.
            block_lengths = np.subtract(session.blocks['end_trials'],
                                        session.blocks['start_trials'])
            block_lengths[0] += residual_trials[ax]
            residual_trials[ax] = block_lengths[-1]
            block_types = {'all': block_lengths[:-1],
                       'neutral': block_lengths[:-1][    np.equal(session.blocks['reward_states'],1)[:-1]],
                   'non_neutral': block_lengths[:-1][np.not_equal(session.blocks['reward_states'],1)[:-1]]}
            day_trials_per_block.append(np.mean(block_types[use_blocks]))
        mean_tpb.append(np.nanmean(day_trials_per_block))
        sd_tpb.append(np.nanstd(day_trials_per_block))
        days_trials_per_block.append(day_trials_per_block)
    days = np.array(list(days))-min(days) + 1
    plt.figure(fig_no)
    if clf: plt.clf()
    plt.subplot(2,1,1)
    plt.errorbar(days, mean_tpb, sd_tpb)
    plt.xlim(0.5, max(days) + 0.5)
    plt.xlabel('Day')
    plt.ylabel('Trials per block')
    print('Mean   trails per block for last {} sessions: {:.1f}'.format(last_n_days, np.mean  (mean_tpb[-last_n_days:])))
    print('Median trails per block for last {} sessions: {:.1f}'.format(last_n_days, np.median(mean_tpb[-last_n_days:])))
    last_n_day_block_lengths = np.array(days_trials_per_block[-last_n_days:]).reshape([1,-1])[0]
    last_n_day_block_lengths = last_n_day_block_lengths[~np.isnan(last_n_day_block_lengths)]
    plt.subplot(2,1,2)
    plt.hist(last_n_day_block_lengths)

def rotation_analysis(sessions, cor_len = 200):
    ''' Evaluate auto and cross correlations for center->side 
    transitions and side->center choices over concatonated sessions for
    each subject.
    '''
    sIDs = list(set([s.subject_ID for s in sessions]))

    sub_trans_auto_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])   # Transition autocorrelations for each subject.
    sub_choice_auto_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])  # Choice (coded clockwise vs anticlockwise) autocorrelation for each subject.
    sub_cross_corrs = np.zeros([len(sIDs), 2 * cor_len + 1])        # Transition - choice cross correlations for each subject.
    for i, sID in enumerate(sIDs):
        a_sessions = sorted([s for s in sessions if s.subject_ID == sID],
                            key = lambda s:s.day)
        choices = []
        transitions = []
        for s in a_sessions:
            transitions += s.CTSO['transitions'][1:].tolist()
            choices += (s.CTSO['choices'][1:] == s.CTSO['second_steps'][:-1]).tolist()

        trans_autocor  = ut.norm_correlate(transitions, transitions)
        choice_autocor = ut.norm_correlate(choices, choices)
        cross_corr     = ut.norm_correlate(transitions, choices)
        cor_inds = list(range(cross_corr.size/2 - cor_len,
                         cross_corr.size/2 + cor_len + 1))
        sub_trans_auto_corrs[i,:] = trans_autocor[cor_inds]
        sub_choice_auto_corrs[i,:] = choice_autocor[cor_inds]
        sub_cross_corrs[i,:] = cross_corr[cor_inds]
    plt.figure(1)
    plt.clf()
    x = list(range(-cor_len, cor_len + 1))
    plt.plot(x, np.mean(sub_trans_auto_corrs, 0))
    plt.plot(x, np.mean(sub_choice_auto_corrs, 0))
    plt.plot(x, np.mean(sub_cross_corrs, 0))
    plt.ylim(0, sorted(np.mean(sub_trans_auto_corrs, 0))[-2])
    plt.legend(('Trans. autocor.', 'Choice autocor.', 'Cross corr.'))
    plt.xlabel('Lag (trials)')
    plt.ylabel('Correlation')

#----------------------------------------------------------------------------------
# Temporal analyses.
#----------------------------------------------------------------------------------

def log_IPI(session):
    'Log inter-poke interval distribution.'
    poke_times = session.ordered_times(['low_poke','high_poke', 'left_poke', 'right_poke'])
    log_IPIs = np.log(poke_times[1::]-poke_times[0:-1])
    bin_edges = np.arange(-1.7,7.6,0.1)
    log_IPI_hist = np.histogram(log_IPIs, bin_edges)[0]
    mode_IPI = np.exp(bin_edges[np.argmax(log_IPI_hist)]+0.05)
    plt.plot(bin_edges[0:-1]+0.05,log_IPI_hist)
    plt.xlim(bin_edges[0],bin_edges[-1])
    plt.xticks(np.log([0.25,1,4,16,64,256]),[0.25,1,4,16,64,256])
    plt.xlabel('Inter-poke interval (sec)')


def log_ITI(sessions, boundary = 10.):
    'log inter - trial start interval distribution.'
    all_ITIs = []
    for session in sessions:
        all_ITIs.append(session.times['trial_start'][1:] - session.times['trial_start'][:-1])
    all_ITIs = np.concatenate(all_ITIs)
    fraction_below_bound = round(sum(all_ITIs < boundary) / len(all_ITIs),3)
    plt.xlim(0,5)
    plt.xlabel('Log inter trial start interval (sec)')
    plt.ylabel('Count')


def reaction_times_second_step(sessions, fig_no = 1):
    'Reaction times for second step pokes as function of common / rare transition.'
    sec_step_IDs = ut.get_IDs(sessions[0].IDs, ['right_active', 'left_active'])
    median_RTs_common = np.zeros(len(sessions))
    median_RTs_rare   = np.zeros(len(sessions))
    for i,session in enumerate(sessions):
        left_active_times = session.times['left_active']
        right_active_times = session.times['right_active']
        left_reaction_times  = _latencies(left_active_times,  session.times['left_poke'])
        right_reaction_times = _latencies(right_active_times, session.times['right_poke'])
        ordered_reaction_times = np.hstack((left_reaction_times,right_reaction_times))\
                                 [np.argsort(np.hstack((left_active_times,right_active_times)))]
        transitions = session.blocks['trial_trans_state'] == session.CTSO['transitions']  # common vs rare.                 
        median_RTs_common[i] = np.median(ordered_reaction_times[ transitions])
        median_RTs_rare[i]    = np.median(ordered_reaction_times[~transitions])
    mean_RT_common = 1000 * np.mean(median_RTs_common)
    mean_RT_rare   = 1000 * np.mean(median_RTs_rare)
    SEM_RT_common = 1000 * sem(median_RTs_common)
    SEM_RT_rare   = 1000 * sem(median_RTs_rare)
    plt.figure(fig_no)
    plt.bar([1,2],[mean_RT_common, mean_RT_rare], yerr = [SEM_RT_common,SEM_RT_rare])
    plt.xlim(0.8,3)
    plt.ylim(mean_RT_common * 0.8, mean_RT_rare * 1.1)
    plt.xticks([1.4, 2.4], ['Common', 'Rare'])
    plt.title('Second step reaction times')
    plt.ylabel('Reaction time (ms)')
    print(('Paired t-test P value: {}'.format(ttest_rel(median_RTs_common, median_RTs_rare)[1])))

def reaction_times_first_step(sessions, fig_no = 1):
    median_reaction_times = np.zeros([len(sessions),4])
    all_reaction_times = []
    for i,session in enumerate(sessions):
        reaction_times = _first_step_RTs(session)
        all_reaction_times.append(reaction_times)
        transitions = (session.blocks['trial_trans_state'] == session.CTSO['transitions'])[:len(reaction_times)] # Transitions common/rare.
        outcomes = session.CTSO['outcomes'][:len(reaction_times)].astype(bool)
        median_reaction_times[i, 0] = np.median(reaction_times[ transitions &  outcomes])  # Common transition, rewarded.
        median_reaction_times[i, 1] = np.median(reaction_times[~transitions &  outcomes])  # Rare transition, rewarded.
        median_reaction_times[i, 2] = np.median(reaction_times[ transitions & ~outcomes])  # Common transition, non-rewarded.
        median_reaction_times[i, 3] = np.median(reaction_times[~transitions & ~outcomes])  # Rare transition, non-rewarded.
    mean_RTs = np.mean(median_reaction_times,0)
    SEM_RTs  = sem(median_reaction_times,0)
    plt.figure(fig_no)
    plt.clf()
    plt.subplot(1,2,1)
    plt.suptitle('First step reaction times')
    plt.bar([1,2,3,4], mean_RTs, yerr = SEM_RTs)
    plt.ylim(min(mean_RTs) * 0.8, max(mean_RTs) * 1.1)
    plt.xticks([1.4, 2.4, 3.4, 4.4], ['Com. Rew.', 'Rare Rew.', 'Com. Non.', 'Rare. Non.'])
    plt.xlim(0.8,5)
    plt.ylabel('Reaction time (ms)')
    all_reaction_times = np.hstack(all_reaction_times)
    cum_rt_hist, bin_edges = _cumulative_histogram(all_reaction_times)
    plt.subplot(1,2,2)
    plt.plot(bin_edges[:-1],cum_rt_hist)
    plt.ylim(0,1)
    plt.xlabel('Time from ITI start (ms)')
    plt.ylabel('Cumumative fraction of first central pokes.')

def _first_step_RTs(session):
    '''Returns array of latencies from ITI start to first central poke 
    starting with second trial (first trial is not preceded by ITI start).'''
    ITI_start_times = session.times['ITI_start']
    center_poke_times = sorted(np.hstack((session.times['high_poke'],
                                          session.times['low_poke'])))
    reaction_times = 1000 * _latencies(ITI_start_times,  center_poke_times)
    reaction_times = reaction_times[:session.n_trials - 1] 
    return reaction_times

def side_poke_out_latencies(sessions, fig_no = 1, peak_scale = 2):
    '''Plot histogram of latencies from entering state wait_for_poke_out untill first
    side poke out.
    '''
    spo_latencies = []
    for session in sessions:
        side_poke_out_times = np.concatenate([session.times['left_poke_out'],
                                              session.times['right_poke_out']])
        spo_latencies.append(1000 * _latencies(session.times['wait_for_poke_out'], 
                                               side_poke_out_times))
    spo_latencies = np.concatenate(spo_latencies)
    bin_edges = np.linspace(np.log(50),np.log(12800),101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    h = np.histogram(np.log(spo_latencies), bin_edges)[0]
    plt.figure(fig_no)
    plt.clf()
    plt.plot(bin_centers,h)
    x_ticks = [50,200,800,3200, 12800]
    plt.xticks(np.log(x_ticks), x_ticks)
    plt.xlabel('Time in side poke (ms)')
    plt.ylabel('Number of trials')
    plt.xlim(np.log([50,12800]))
    ymax = plt.ylim()[1]
    if peak_scale: # Find peaks of specified scale
        peaks = find_peaks_cwt(h,np.array([peak_scale]))
        for pk in peaks:
            plt.plot([bin_centers[pk], bin_centers[pk]],[0,ymax],'k')
            print('Peak at :{} ms'.format(int(np.exp(bin_centers[pk]))))


def _cumulative_histogram(data, bin_edges = np.arange(0,3001)):
    h = np.histogram(data, bin_edges)[0]
    cum_hist = np.cumsum(h) / len(data)
    return cum_hist, bin_edges


def trials_per_minute(sessions, smooth_SD = 5, ses_dur = 120, fig_no = 1, ebars = 'SEM',
                      clf = True, col = 'b', plot_cum = False):
    bin_edges = np.arange(ses_dur + 1)
    all_trials_per_minute = np.zeros((len(sessions),ses_dur))
    for i, session in enumerate(sessions):
        trials_per_min = np.histogram(session.times['trial_start']/60, bin_edges)[0]
        if smooth_SD: #Smooth by convolution with gaussian of specified standard deviation.
            trials_per_min = gaussian_filter1d(trials_per_min, smooth_SD)
        all_trials_per_minute[i,:] = trials_per_min
    mean_tpm = np.mean(all_trials_per_minute,0)
    sd_tpm = np.std(all_trials_per_minute,0)
    sem_tpm = sd_tpm / np.sqrt(len(sessions))
    cumulative_tpm = np.cumsum(all_trials_per_minute,1)
    mean_ctpm = np.mean(cumulative_tpm,0)
    sd_ctpm = np.std(cumulative_tpm,0)
    sem_ctpm = sd_ctpm / np.sqrt(len(sessions))    
    plt.figure(fig_no)
    if clf: plt.clf()
    if plot_cum:
        plt.subplot(2,1,1)
    plt.plot(bin_edges[1:], mean_tpm, color = col)
    if ebars == 'SD':
        plt.fill_between(bin_edges[1:], mean_tpm-sd_tpm, mean_tpm+sd_tpm, alpha = 0.2, facecolor = col)
    elif ebars == 'SEM':
        plt.fill_between(bin_edges[1:], mean_tpm-sem_tpm, mean_tpm+sem_tpm, alpha = 0.2, facecolor = col)
    plt.ylabel('Trials per minute') 
    plt.xlim(1,ses_dur)
    plt.ylim(ymin = 0)
    if plot_cum:
        plt.subplot(2,1,2)
        plt.plot(bin_edges[1:], mean_ctpm, color = col)
        if ebars == 'SD':
            plt.fill_between(bin_edges[1:], mean_ctpm-sd_ctpm, mean_ctpm+sd_ctpm, alpha = 0.2, facecolor = col)
        elif ebars == 'SEM':
            plt.fill_between(bin_edges[1:], mean_ctpm-sem_ctpm, mean_ctpm+sem_ctpm, alpha = 0.2, facecolor = col)
        plt.ylabel('Cum. trials per minute') 
        plt.xlabel('Time (mins)')    
        plt.xlim(1,ses_dur)
        plt.ylim(ymin = 0)
    else:
        plt.xlabel('Time (mins)')  

def ITI_poke_timings(sessions, fig_no = 1):
    'Plots the timing of central pokes relative to the inter-trail interval, averaged over multiple sessions.'
    center_poke_TS_hists,  first_poke_TS_hists,  ITI_poke_TS_hists, \
    center_poke_ITI_hists, first_poke_ITI_hists, ITI_poke_ITI_hists, cum_fp_hists = ([], [], [], [], [], [], [])
    for session in sessions:
        center_poke_TS_hist,  first_poke_TS_hist,  ITI_poke_TS_hist, \
        center_poke_ITI_hist, first_poke_ITI_hist, ITI_poke_ITI_hist, cum_fp_hist, bin_edges = _ITI_analysis(session)
        center_poke_TS_hists.append(center_poke_TS_hist)
        first_poke_TS_hists.append(first_poke_TS_hist)
        ITI_poke_TS_hists.append(ITI_poke_TS_hist)
        first_poke_ITI_hists.append(first_poke_ITI_hist)
        center_poke_ITI_hists.append(center_poke_ITI_hist)
        ITI_poke_ITI_hists.append(ITI_poke_ITI_hist)
        cum_fp_hists.append(cum_fp_hist)
    print('Fraction ITI without poke: {}'.format(np.mean([successful_delay_fraction(s) for s in sessions])))
    plt.figure(fig_no)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    plt.clf()
    plt.subplot(3,1,1)
    plt.fill_between(bin_centers, np.mean(center_poke_TS_hists, 0) / np.sum(first_poke_TS_hists), color = 'b')
    plt.fill_between(bin_centers, np.mean(first_poke_TS_hists,  0) / np.sum(first_poke_TS_hists), color = 'g')
    plt.xlim(bin_centers[0], bin_centers[-1])
    plt.ylim(ymin = 0)
    plt.xlabel('Time relative to trial start (sec)')
    plt.subplot(3,1,2)
    plt.fill_between(bin_centers, np.mean(center_poke_ITI_hists, 0) / np.sum(first_poke_TS_hists), color = 'b')
    plt.fill_between(bin_centers, np.mean(first_poke_ITI_hists,  0) / np.sum(first_poke_TS_hists), color = 'g')
    plt.xlim(bin_centers[0], bin_centers[-1])
    plt.ylim(ymin = 0)
    plt.xlabel('Time relative to ITI start (sec)')
    plt.subplot(3,2,5)
    plt.plot(bin_centers, np.mean(cum_fp_hists, 0))
    plt.xlim(0,bin_centers[-1])
    plt.ylim(0,1)
    plt.xlabel('Reaction time (sec)')
    plt.ylabel('Fraction of trials')

def _ITI_analysis(session, max_delta_t = 2.5, resolution = 0.01):
    'Evaluates the timing of central pokes relative to the inter-trail interval for single session.'
    assert 'ITI_start' in list(session.IDs.keys()), 'Session does not have inter-trial interval'
    # Get time stamps for relevent events
    center_pokes = session.ordered_times(['high_poke','low_poke'])
    trial_starts = session.times['trial_starts']
    ITI_starts   = session.times['ITI_starts']
    first_pokes_of_trial = []
    for trial_start in trial_starts[:-1]:
        first_pokes_of_trial.append(center_pokes[np.argmax(center_pokes > trial_start)])
    pokes_during_ITI  = np.array([])
    ITI_poke_delta_ts = np.array([])
    for ITI_start, trial_start in zip(ITI_starts, trial_starts[1:]):
        pokes_during_this_ITI = center_pokes[(center_pokes > ITI_start) & (center_pokes < trial_start)]
        pokes_during_ITI = np.append(pokes_during_ITI, pokes_during_this_ITI)
        ITI_poke_delta_ts = np.append(ITI_poke_delta_ts, pokes_during_this_ITI[:1] - ITI_start)
    # histograms with respect to trial start.
    center_poke_TS_hist, bin_edges = _PETH(center_pokes, trial_starts, max_delta_t, resolution) #PETH of center pokes wrt trial starts.
    first_poke_TS_delta_ts = first_pokes_of_trial - trial_starts[:len(first_pokes_of_trial)]
    first_poke_TS_hist = np.histogram(first_poke_TS_delta_ts, bin_edges)[0]                     #PETH of first poke of trial wrt trial start.
    ITI_poke_TS_hist = _PETH(pokes_during_ITI, trial_starts, max_delta_t, resolution)[0]        #PETH of pokes during ITI wrt trial start.
    # histograms with respect to ITI start.
    center_poke_ITI_hist  = _PETH(center_pokes, ITI_starts, max_delta_t, resolution)[0]         #PETH of center pokes wrt ITI starts.
    first_poke_ITI_delta_ts = first_pokes_of_trial[1:] - ITI_starts[:len(first_pokes_of_trial)-1]
    first_poke_ITI_hist = np.histogram(first_poke_ITI_delta_ts, bin_edges)[0]                   #PETH of first poke of trial wrt ITI start.
    ITI_poke_ITI_hist = np.histogram(ITI_poke_delta_ts, bin_edges)[0]                           #PETH of pokes during ITI wrt ITI start.
    # Cumulative histogram of first pokes relative to trial start.    
    cum_fp_hist = np.cumsum(first_poke_ITI_hist) / session.n_trials
    return (center_poke_TS_hist,  first_poke_TS_hist,  ITI_poke_TS_hist, \
            center_poke_ITI_hist, first_poke_ITI_hist, ITI_poke_ITI_hist, cum_fp_hist, bin_edges)

def _latencies(event_times_A, event_times_B):
    'Evaluate the latency between each event A and the first event B that occurs afterwards.'                
    latencies = np.outer(event_times_B, np.ones(len(event_times_A))) - \
                np.outer(np.ones(len(event_times_B)), event_times_A)
    latencies[latencies <= 0] = np.inf
    latencies = np.min(latencies,0)
    return latencies

def _PETH(time_stamps_1, time_stamps_2, max_delta_t, resolution):
    'Peri-event time histogram of event 1 relative to event 2.'
    delta_ts = np.tile(time_stamps_1,(len(time_stamps_2),1)).T - \
               np.tile(time_stamps_2,(len(time_stamps_1),1))
    delta_ts = delta_ts = delta_ts[abs(delta_ts) < max_delta_t]
    return np.histogram(delta_ts, np.arange(-max_delta_t, max_delta_t + resolution, resolution))
    
def successful_delay_fraction(session):
    'Evaluates fraction of trials on which animal witholds central poking during ITI'
    assert 'ITI_start' in list(session.IDs.keys()), 'Session does not have inter-trial interval'
    event_IDs =  ut.get_IDs(session.IDs, ['low_poke','high_poke', 'trial_start', 'ITI_start'])
    ev_seq = [ev for ev in session.event_codes if ev in event_IDs]
    ev_seq.append(-1) # To avoid index out of range errors at next line.
    events_following_ITI_start = [ev_seq[i+1] for i, x in enumerate(ev_seq) if x == session.IDs['ITI_start']]
    return np.mean(np.array(events_following_ITI_start) == session.IDs['trial_start'])

#----------------------------------------------------------------------------------
# Choice probability trajectory analyses.
#----------------------------------------------------------------------------------

def reversal_analysis(sessions, pre_post_trials = [-15,40], last_n = 15, fig_no = 3,
                      return_fits = False, clf = True, cols = 0, by_type = True):
    '''Analysis of choice trajectories around reversals in reward probability and
    transition proability.  Fits exponential decay to choice trajectories following reversals.'''
    if len(set(np.hstack([s.blocks['transition_states'] for s in sessions]))) == 1:
        by_type = False # Can't evaluate by reversal type if data has no transition reversals.

    p_1 = _end_of_block_p_correct(sessions, last_n)
    if by_type: # Analyse reversals in reward and transition probabilities seperately.
        choice_trajectories_rr = _get_choice_trajectories(sessions, 'reward_reversal', pre_post_trials)
        fit_rr = _fit_exp_to_choice_traj(choice_trajectories_rr, p_1, pre_post_trials, last_n)
        choice_trajectories_tr = _get_choice_trajectories(sessions, 'reward_unchanged', pre_post_trials)
        fit_tr = _fit_exp_to_choice_traj(choice_trajectories_tr, p_1,pre_post_trials, last_n)
    else:
        fit_rr, fit_tr = (None, None)

    choice_trajectories_br = _get_choice_trajectories(sessions, 'any_reversal', pre_post_trials)
    fit_br = _fit_exp_to_choice_traj(choice_trajectories_br, p_1, pre_post_trials, last_n)

    if return_fits:
        return {'p_1'      :p_1,
                'rew_rev'  :fit_rr,
                'trans_rev':fit_tr,
                'both_rev' :fit_br}
    else:
        colors = (('c','b'),('y','r'))[cols]
        plt.figure(fig_no)
        if clf:plt.clf()   
        if by_type:
            plt.subplot(3,1,1)
            plt.title('Reversal in reward probabilities', fontsize = 'small')
            _plot_mean_choice_trajectory(choice_trajectories_rr, pre_post_trials, colors[0])
            _plot_exponential_fit(fit_rr, p_1, pre_post_trials, last_n, colors[1])
            plt.subplot(3,1,2)
            plt.title('Reversal in transition probabilities', fontsize = 'small')
            _plot_mean_choice_trajectory(choice_trajectories_tr, pre_post_trials, colors[0])
            _plot_exponential_fit(fit_tr, p_1, pre_post_trials, last_n, colors[1])
            plt.subplot(3,1,3)
            plt.title('Both reversals combined', fontsize = 'small')
        _plot_mean_choice_trajectory(choice_trajectories_br, pre_post_trials, colors[0])
        _plot_exponential_fit(fit_br, p_1, pre_post_trials, last_n, colors[1])
        plt.xlabel('Trials relative to block transition.')
        plt.ylabel('Fraction of choices to pre-reversal correct side.')
        print(('Average block end choice probability: {}'.format(p_1)))
        if by_type:
            print(('Reward probability reversal, tau: {}, P_0: {}'.format(fit_rr['tau'], fit_rr['p_0'])))
            print(('Trans. probability reversal, tau: {}, P_0: {}'.format(fit_tr['tau'], fit_tr['p_0'])))
        print(('Combined reversals,          tau: {}, P_0: {}'.format(fit_br['tau'], fit_br['p_0'])))

def _block_index(blocks):
    '''Create dict of boolean arrays used for indexing block transitions,
    Note first value of index corresponds to second block of session.'''
    return {
    'transition_reversal' : np.array(blocks['transition_states'][:-1]) != np.array(blocks['transition_states'][1:]),
    'to_neutral'          : np.array(blocks['reward_states'][1:]) == 1,
    'from_neutral'        : np.array(blocks['reward_states'][:-1]) == 1,
    'reward_reversal'     : np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2,
    'reward_unchanged'    : np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]),
    'any_reversal'        : (np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2) | \
                            (np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]))}

def _get_choice_trajectories(sessions, trans_type, pre_post_trials):
    '''Evaluates choice trajectories around transitions of specified type. Returns float array
     of choice trajectories of size (n_transitions, n_trials). Choices are coded such that a 
    choice towards the option which is correct before the transition is 1, the other choice is 0,
    if the choice trajectory extends past the ends of the blocks before and after the transition
    analysed, it is padded with nans.'''
    choice_trajectories = []
    n_trans_analysed = 0
    n_trials = pre_post_trials[1] - pre_post_trials[0]
    for session in sessions:
        blocks = session.blocks
        selected_transitions = _block_index(blocks)[trans_type]
        n_trans_analysed +=sum(selected_transitions)
        start_trials = np.array(blocks['start_trials'][1:])[selected_transitions] # Start trials of blocks following selected transitions.
        end_trials = np.array(blocks['end_trials'][1:])[selected_transitions]     # End trials of blocks following selected transitions.
        prev_start_trials = np.array(blocks['start_trials'][:-1])[selected_transitions] # Start trials of blocks preceding selected transitions.
        transition_states = np.array(blocks['transition_states'][:-1])[selected_transitions] # Transition state of blocks following selected transitions.
        reward_states = np.array(blocks['reward_states'][:-1])[selected_transitions] # Reward state of blocks following selected transitions.

        for     start_trial,  end_trial,  prev_start_trial,  reward_state,  transition_state in \
            zip(start_trials, end_trials, prev_start_trials, reward_states, transition_states):

            trial_range = start_trial + np.array(pre_post_trials)
            if trial_range[0] < prev_start_trial:
                pad_start = prev_start_trial - trial_range[0] 
                trial_range[0] = prev_start_trial
            else:
                pad_start = 0
            if trial_range[1] > end_trial:
                pad_end = trial_range[1] - end_trial
                trial_range[1] = end_trial
            else:
                pad_end = 0
            choice_trajectory = session.CTSO['choices'][trial_range[0]:trial_range[1]].astype(bool)                        
            choice_trajectory = (choice_trajectory ^ bool(reward_state) ^ bool(transition_state)).astype(float)
            if pad_start:
                choice_trajectory = np.hstack((ut.nans(pad_start), choice_trajectory))
            if pad_end:
                choice_trajectory = np.hstack((choice_trajectory, ut.nans(pad_end)))
            choice_trajectories.append(choice_trajectory)
    return np.vstack(choice_trajectories)

def _plot_mean_choice_trajectory(choice_trajectories, pre_post_trials, col = 'b'):
    plt.plot(list(range(pre_post_trials[0], pre_post_trials[1])), np.nanmean(choice_trajectories,0),col)
    plt.plot([0,0],[0,1],'k--')
    plt.plot([pre_post_trials[0], pre_post_trials[1]-1],[0.5,0.5],'k:')
    plt.ylim(0,1)
    plt.xlim(pre_post_trials[0],pre_post_trials[1])

def _plot_exponential_fit(fit, p_1, pre_post_trials, last_n, col = 'r'):
    t = np.arange(0,pre_post_trials[1])
    p_traj = np.hstack([ut.nans(-pre_post_trials[0]-last_n), np.ones(last_n) * fit['p_0'], \
                   _exp_choice_traj(fit['tau'], fit['p_0'], p_1, t)])
    plt.plot(list(range(pre_post_trials[0], pre_post_trials[1])),p_traj, col, linewidth = 2)
    plt.locator_params(nbins = 4)

def _end_of_block_p_correct(sessions, last_n = 15, stim_type = None, return_n_trials = False):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks.'
    n_correct, n_trials = (0, 0)
    for session in sessions:
        if last_n == 'all':  # Use all trials in block non neutral blocks.
            block_end_trials = session.select_trials('all', block_type = 'non_neutral')
        else:  # Use only last_n  trials of non neutral blocks. 
            block_end_trials = session.select_trials('end', last_n, block_type = 'non_neutral')
        if stim_type:
            assert stim_type in ['stim', 'non_stim'], 'Invalid stim_type argument.'
            if stim_type == 'stim':
                block_end_trials = block_end_trials &  session.stim_trials
            elif stim_type == 'non_stim':
                block_end_trials = block_end_trials & ~session.stim_trials
        n_trials += sum(block_end_trials)
        correct_choices = session.CTSO['choices'] ^ \
                          np.array(session.blocks['trial_rew_state'],   bool) ^ \
                          np.array(session.blocks['trial_trans_state'], bool)
        n_correct += sum(correct_choices[block_end_trials])
    p_correct = n_correct / n_trials
    if return_n_trials:
        return p_correct, n_trials
    else:
        return p_correct

def _fit_exp_to_choice_traj(choice_trajectories, p_1, pre_post_trials,  last_n):
    '''Fit an exponential curve to the choice trajectroy following a block transition
    using maximum likelihood.  The only parameter that is adjusted is the time constant,
    the starting value is determined by the mean choice probability in the final last_n trials
    before the transition, and the asymptotic choice  probability is given by  (1 - p_1).
    '''

    n_traj = np.sum(~np.isnan(choice_trajectories),0)  # Number of trajectories at each timepoint.
    n_post = n_traj[-pre_post_trials[1]:]  # Section folowing transtion.
    if min(n_traj) == 0:
        return {'p_0':np.nan,'tau':np.nan}
    t = np.arange(0,pre_post_trials[1])
    sum_choices = np.nansum(choice_trajectories, 0)
    p_0 = (np.sum(sum_choices[-pre_post_trials[1]-last_n:-pre_post_trials[1]]) /  # Choice probability at end of previous block.
           np.sum(     n_traj[-pre_post_trials[1]-last_n:-pre_post_trials[1]]))
    q = sum_choices[-pre_post_trials[1]:]  # Number of choices to previously correct side at different timepoints relative to block transition.
    fits = np.zeros([3,2])
    for i, x_0 in enumerate([10.,20.,40.]):   # Multiple starting conditions for minimization. 
        minimize_output = minimize(_choice_traj_likelihood, np.array([x_0]),
                                    method = 'Nelder-Mead', args = (p_0, p_1, q, n_post, t))
        fits[i,0] = minimize_output['x'][0]
        fits[i,1] = minimize_output['fun']
    tau_est = fits[0,np.argmin(fits[1,:])] # Take fit with highest likelihood.
    return {'p_0':p_0,'tau':tau_est}

def _choice_traj_likelihood(tau, p_0, p_1, q, n, t):
    if tau < 0: return np.inf
    p_traj = _exp_choice_traj(tau, p_0, p_1, t)
    log_lik = binom.logpmf(q,n,p_traj).sum()
    return -log_lik

def _exp_choice_traj(tau, p_0, p_1, t):
    return (1. - p_1) + (p_0 + p_1 - 1.) * np.exp(-t/tau)

def per_animal_end_of_block_p_correct(sessions, last_n = 15, fig_no = 1, col = 'b', clf = True):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks on a per animals basis.'
    p_corrects = []
    for sID in sorted(set([s.subject_ID for s in sessions])):
        p_corrects.append(_end_of_block_p_correct([s for s in sessions if s.subject_ID == sID]))
    plt.figure(fig_no)
    if clf: plt.clf()
    n_sub = len(p_corrects)
    plt.scatter(0.2*np.random.rand(n_sub),p_corrects, s = 8,  facecolor= col, edgecolors='none', lw = 0)
    plt.errorbar(0.1, np.mean(p_corrects), np.std(p_corrects),linestyle = '', marker = '', linewidth = 2, color = col)
    plt.xlim(-1,1)
    plt.xticks([])
    plt.ylabel('Prob. correct choice')
    return p_corrects


def session_start_analysis(sessions, first_n = 40):
    'Analyses choice trajectories following session start.'
    choice_trajectories = []
    for session in sessions:
        reward_state = session.blocks['reward_states'][0]
        if not reward_state == 1: # don't analyse sessions that start in neutral block. 
            transition_state = session.blocks['transition_states'][0]
            choice_trajectory = session.CTSO['choices'][:first_n]
            choice_trajectory = choice_trajectory ^ bool(reward_state) ^ bool(transition_state)  
            choice_trajectories.append(choice_trajectory)
    plt.figure(1)
    plt.clf()
    plt.plot(np.mean(choice_trajectories,0))
    plt.xlabel('Trial number')
    plt.ylabel('Choice Probability')


#----------------------------------------------------------------------------------
# Stay probability Analysis
#----------------------------------------------------------------------------------

def stay_probability_analysis(sessions, ebars = 'SEM', selection = 'xtr',
                              fig_no = 1, ymin = 0, trial_mask = None):
    '''Stay probability analysis for task version with reversals in transition matrix.
    '''
    assert ebars in [None, 'SEM', 'SD'], 'Invalid error bar specifier.'
    n_sessions = len(sessions)
    all_n_trials, all_n_stay = (np.zeros([n_sessions,12]), np.zeros([n_sessions,12]))
    for i, session in enumerate(sessions):
        trial_select = session.select_trials(selection)
        if trial_mask is not None:
            trial_select = trial_select & trial_mask[i]
        trial_select_A = trial_select &  session.blocks['trial_trans_state']
        trial_select_B = trial_select & ~session.blocks['trial_trans_state']
        #Eval total trials and number of stay trial for A and B blocks.
        all_n_trials[i,:4], all_n_stay[i,:4] = _stay_prob_analysis(session.CTSO, trial_select_A)
        all_n_trials[i,4:8], all_n_stay[i,4:8] = _stay_prob_analysis(session.CTSO, trial_select_B)
        # Evaluate combined data.
        all_n_trials[i,8:] = all_n_trials[i,:4] + all_n_trials[i,[5,4,7,6]]
        all_n_stay[i,8:] = all_n_stay[i,:4] + all_n_stay[i,[5,4,7,6]]
    if not ebars: # Don't calculate cross-animal error bars.
        mean_stay_probs = np.nanmean(all_n_stay / all_n_trials, 0)
        y_err  = np.zeros(12)
    else:
        session_sIDs = np.array([s.subject_ID for s in sessions])
        unique_sIDs = list(set(session_sIDs))
        n_subjects = len(unique_sIDs)
        per_subject_stay_probs = np.zeros([n_subjects,12])
        for i, sID in enumerate(unique_sIDs):
            session_mask = session_sIDs == sID # True for sessions with correct animal ID.
            per_subject_stay_probs[i,:] = sum(all_n_stay[session_mask,:],0) / sum(all_n_trials[session_mask,:],0)
        mean_stay_probs = np.nanmean(per_subject_stay_probs, 0)
        if ebars == 'SEM':
            y_err = ut.nansem(per_subject_stay_probs, 0)
        else:
            y_err = np.nanstd(per_subject_stay_probs, 0)

    if fig_no:
        plt.figure(fig_no)
        plt.clf()
        plt.subplot(1,3,1)
        plt.bar(np.arange(1,5), mean_stay_probs[:4], yerr = y_err[:4])
        plt.ylim(ymin,1)
        plt.xlim(0.75,5)
        plt.title('A transitions normal.', fontsize = 'small')
        plt.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
        plt.ylabel('Stay Probability')
        plt.subplot(1,3,2)
        plt.bar(np.arange(1,5), mean_stay_probs[4:8], yerr = y_err[4:8])
        plt.ylim(ymin,1)
        plt.xlim(0.75,5)
        plt.title('B transitions normal.', fontsize = 'small')
        plt.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
        plt.subplot(1,3,3)
        plt.bar(np.arange(1,5), mean_stay_probs[8:], yerr = y_err[8:])
        plt.ylim(ymin,1)
        plt.xlim(0.75,5)
        plt.title('Combined.', fontsize = 'small')
        plt.xticks([1.5,2.5,3.5,4.5],['1/N', '1/R', '0/N', '0/R'])
    else:
        return mean_stay_probs, y_err

def _stay_prob_analysis(CTSO, trial_select):
    'Analysis for stay probability plots using binary mask to select trials.'
    choices, transitions, outcomes = ut.CTSO_unpack(CTSO, 'CTO', bool)
    stay = choices[1:] == choices[:-1]
    transitions, outcomes, trial_select = (transitions[:-1], outcomes[:-1], trial_select[:-1])
    stay_go_by_type = [stay[( outcomes &  transitions) & trial_select],  # A transition, rewarded.
                       stay[( outcomes & ~transitions) & trial_select],  # B transition, rewarded.
                       stay[(~outcomes &  transitions) & trial_select],  # A transition, not rewarded.
                       stay[(~outcomes & ~transitions) & trial_select]]  # B transition, not rewarded.
    n_trials_by_type = [len(s) for s in stay_go_by_type]
    n_stay_by_type =   [sum(s) for s in stay_go_by_type]
    return n_trials_by_type, n_stay_by_type

#----------------------------------------------------------------------------------
# Functions called by session and experiment classes.
#----------------------------------------------------------------------------------

def plot_day(exp, day):
    if day < 0: day = exp.n_days + day + 1
    day_sessions = exp.get_sessions('all', days = day)
    plt.figure(day)
    for i, session in enumerate(day_sessions):
        plt.subplot(len(day_sessions),1,i+1)
        session_plot(session, ylabel = False)
        plt.ylabel(session.subject_ID)
    plt.suptitle('Day number: {}, Date: '.format(session.day) + session.date)
   
def plot_subject(exp, sID, day_range = [0, np.inf]):
    subject_sessions =  exp.get_sessions(sID, 'all')
    if hasattr(subject_sessions[0], 'day'):
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.day)
        sorted_sessions = [s for s in sorted_sessions if
                           s.day >= day_range[0] and s.day <= day_range[1]]
    else:
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.number)
    n_sessions = len(sorted_sessions)
    plt.figure(sID)
    for i,session in enumerate(sorted_sessions):
        plt.subplot(n_sessions, 1, i+1)
        session_plot(session, ylabel = False)
        plt.ylabel(session.subject_ID)
        if hasattr(session, 'day'):
            plt.ylabel(session.day)
        else:
            plt.ylabel(session.number)
    plt.suptitle('Subject ID'.format(session.subject_ID))

def plot_session(session, fig_no = 1):
    'Plot data from a single session.'
    plt.figure(fig_no)
    plt.clf()
    session_plot(session)

