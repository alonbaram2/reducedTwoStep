import os
import pickle
import imp
import numpy as np
from . import session as ss
from . import plotting as pl

class experiment:
    def __init__(self, exp_name, rebuild_sessions=False, missing_data_warning=False, exp_path=None):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from subjects in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.start_date = exp_name[:10]  

        if exp_path:
            self.path = exp_path
        else:
            self.path = os.path.join('..', 'data sets', exp_name)

        self.data_path = os.path.join(self.path, 'data')

        info = imp.load_source('info', os.path.join(self.path, 'info.py'))

        self.IDs = info.IDs
        self.info = info.info
        self.file_type = info.file_type

        assert self.file_type in ['Arduino', 'pyControl_1', 'pyControl_2'], 'Invalid file type.'
        
        required_IDs = {'low_poke', 'high_poke', 'left_poke', 'right_poke', 'trial_start', 'left_reward',
                        'right_reward','left_active','right_active','ITI_start', 'wait_for_poke_out'}

        assert required_IDs <= self.IDs.keys(), 'IDs dictionary missing keys: ' + repr(list(required_IDs - set(self.IDs.keys())))

        self.sessions = []
        
        if not rebuild_sessions:
            try:
                with open(os.path.join(self.path, 'sessions.pkl'),'rb') as f:
                    self.sessions = pickle.load(f)
                print('Saved sessions loaded from: sessions.pkl')
            except IOError:
               pass

        self.import_data()

        if missing_data_warning: self.check_for_missing_data_files()
        if rebuild_sessions: self.save()

    def save(self):
        'Save sessions from experiment.'
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as f:
            pickle.dump(self.sessions, f)

    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        with open(os.path.join(self.path, file_name + '.pkl'), 'wb') as f:
            pickle.dump(item, f)

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        with open(os.path.join(self.path, item_name + '.pkl'), 'rb') as f:
                return pickle.load(f)

    def import_data(self):
        '''Load new sessions as session class instances.'''

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.data_path)
        new_files = [f for f in files if f[0] == 'm' and f not in old_files]

        if len(new_files) > 0:
            print('Loading new data files...')
            new_sessions = []
            for file_name in new_files:
                try:
                    new_sessions.append(ss.session(file_name,self.data_path, self.IDs, self.file_type))
                except AssertionError as error_message:
                    print('Unable to import file: ' + file_name)
                    print(error_message)

            self.sessions = self.sessions + new_sessions  

        self.dates = sorted(list(set([session.date for session in self.sessions])))

        for session in self.sessions: # Assign day numbers.
            session.day = self.dates.index(session.date) + 1

        self.n_subjects = len(set([session.subject_ID for session in self.sessions]))
        self.n_days = max([session.day for session in self.sessions]) 
        self.subject_IDs= list(set([s.subject_ID for s in self.sessions]))

        
    def get_sessions(self, sIDs, days = [], dates = []):
        '''Return list of sessions which match specified subject ID and day numbers
        or dates. 
        Select all days or subjects with:  days = 'all', sIDs = 'all'
        Select the last n days with     :  days = -n. 
        Select days from n to end with  :  days = [n, -1]
        '''
        if days == 'all':
            days = range(self.n_days + 1)
        elif isinstance(days, int):
            if days < 0:
                days = list(range(self.n_days + 1 + days, self.n_days + 1))
            else: days = [days]
        elif len(days) == 2 and days[-1] == -1:
            days = range(days[0], self.n_days + 1)
        if sIDs == 'all':
            sIDs = self.subject_IDs
        elif isinstance(sIDs, int):
            sIDs = [sIDs]
        valid_sessions = [s for s in self.sessions if 
            (s.day in days or s.date in dates) and s.subject_ID in sIDs]
        if len(valid_sessions) == 0:
            return None
        elif len(valid_sessions) == 1: 
            return valid_sessions[0] # Don't return list for single session.
        else:
            return valid_sessions                
                 
    def print_CSO_to_file(self, sIDs, days, file_name = 'sessions_CSO.txt'):
        f = open(file_name, 'w')
        sessions = self.get_sessions(sIDs, days)
        total_trials = sum([s.n_trials for s in sessions])
        f.write('Data from experiment "{}", {} sessions, {} trials.\n' 
                'Each trial is indicated by 3 numbers:\n'
                'First column : Choice      (1 = high poke, 0 = low poke)\n'
                'Second column: Second step (1 = left poke, 0 = right poke)\n'
                'Third column : Outcome     (1 = rewarded, 0 = not rewarded)\n'
                .format(self.name, len(sessions), total_trials))
        for (i,s) in enumerate(sessions):
            f.write('''\nSession: {0}, subject ID: {1}, date: {2}\n\n'''\
                    .format(i + 1, s.subject_ID, s.date))
            for c,sl,o in zip(s.trial_data['choices'], s.trial_data['second_links'], s.trial_data['outcomes']):
                f.write('{0:1d} {1:1d} {2:1d}\n'.format(c, sl, o))
        f.close()

    def check_for_missing_data_files(self):
        '''Identifies any days where there are data files for only a subset of subjects
        and reports missing sessions. Called on instantiation of experiment as a check 
        for any problems in the date transfer pipeline from rig to analysis.
        '''
        dates = sorted(set([s.date for s in self.sessions]))
        sessions_per_date = [len(self.get_sessions('all', dates = date)) for date in dates]
        if min(sessions_per_date) < self.n_subjects:
            print('Possible missing data files:')
            for date, n_sessions in zip(dates, sessions_per_date):
                if n_sessions < self.n_subjects:
                    subjects_run = [s.subject_ID for s in self.get_sessions('all', dates = date)]
                    subjects_not_run = set(self.subject_IDs) - set(subjects_run)
                    for sID in subjects_not_run:
                        print(('Date: ' + date + ' sID: {}'.format(sID)))

    def concatenate_sessions(self, days):
        ''' For each subject, concatinate sessions for specified days
        into single long sessions.
        '''
        concatenated_sessions = []
        for sID in self.subject_IDs:
            subject_sessions = self.get_sessions(sID, days)
            concatenated_sessions.append(ss.concatenated_session(subject_sessions))
        return concatenated_sessions

    # Plotting.

    def plot_day(self, day = -1): pl.plot_day(self, day)
    def plot_subject(self, sID, day_range = [0, np.inf]): pl.plot_subject(self, sID, day_range) 




