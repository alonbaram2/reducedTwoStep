 # Class containing data from all sessions in one experiment.

import os
import pickle
import imp
import scipy.io as sio
import numpy as np
from . import session as ss
from . import utility as ut
from . import plotting as pl

class experiment:
    def __init__(self, exp_name, rebuild_sessions = False):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from subjects in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.start_date = exp_name[:10]  

        self.path = os.path.join('..', 'data sets', exp_name)

        self.data_path = os.path.join(self.path, 'data')

        info = imp.load_source('info', os.path.join(self.path, 'info.py'))

        self.IDs = info.IDs
        self.info = info.info
   
        self.sessions = []
        
        if not rebuild_sessions:
            try:
                exp_file = open(os.path.join(self.path, 'sessions.pkl'),'rb')
                self.sessions = pickle.load(exp_file)
                exp_file.close()
                print('Saved sessions loaded from: sessions.pkl')
            except IOError:
               pass

        self.import_data()
        self.check_for_missing_data_files()

        if rebuild_sessions:
            self.save()

    def save(self):
        'Save sessions from experiment.'
        exp_file = open(os.path.join(self.path, 'sessions.pkl'),'wb')
        pickle.dump(self.sessions, exp_file)
        exp_file.close()

    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        f = open(self.path + file_name + '.pkl', 'wb')
        pickle.dump(item, f)
        f.close()

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        f = open(self.path + item_name + '.pkl', 'rb')
        out = pickle.load(f)
        f.close()
        return out

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
                    new_sessions.append(ss.session(file_name,self.data_path,self.IDs))
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
        or dates. All days or subjects can be selected with input 'all'.
        The last n days can be selected with days = -n .
        '''
        if days == 'all':
            days = np.arange(self.n_days) + 1
        elif isinstance(days, int):
            if days < 0:
                days = list(range(self.n_days + 1 + days, self.n_days + 1))
            else: days = [days]
        if sIDs == 'all':
            sIDs = self.subject_IDs
        elif isinstance(sIDs, int):
            sIDs = [sIDs]
        valid_sessions = [s for s in self.sessions if 
            (s.day in days or s.date in dates) and s.subject_ID in sIDs]
        if len(valid_sessions) == 1: 
            valid_sessions = valid_sessions[0] # Don't return list for single session.
        return valid_sessions                
                 
    def print_CSO_to_file(self, sIDs, days, file_name = 'sessions_CSO.txt'):
        f = open(file_name, 'w')
        sessions = self.get_sessions(sIDs, days)
        total_trials = sum([s.n_trials for s in sessions])
        f.write('''Data from experiment "{0}", {1} sessions, {2} trials.\n 
Each trial is indicated by 3 numbers:
First column : Choice      (1 = high poke, 0 = low poke)
Second column: Second step (1 = left poke, 0 = right poke)
Third column : Outcome     (1 = rewarded, 0 = not rewarded)\n''' \
            .format(self.name, len(sessions), total_trials))
        for (i,s) in enumerate(sessions):
            f.write('''\nSession: {0}, subject ID: {1}, date: {2}\n\n'''\
                    .format(i + 1, s.subject_ID, s.date))
            for c,sl,o in zip(s.CTSO['choices'], s.CTSO['second_links'], s.CTSO['outcomes']):
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




