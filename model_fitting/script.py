import data_import as di
import model_fitting as mf
import MF_MB_agent

data_path = 'C:\\Users\\Thomas\\Dropbox\\Work\\Human two step\\Alon repo\\bhv_data'

session = di.load_subject(2, data_path)

early = session.loc[session['session_n'].isin(range(1,5))]
mid   = session.loc[session['session_n'].isin(range(5,9))]
late  = session.loc[session['session_n'].isin(range(9,13))]

fit_early = mf.fit_session(early, MF_MB_agent, repeats=10, use_prior=True)