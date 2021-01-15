from Two_step import *

agent_lr = lr.config_log_reg([ 'correct','choice','outcome','trans_CR', 'trCR_x_out'])

#-----------------------------------------------------
# New data
#-----------------------------------------------------

exp_smpl = he.experiment('2015-10-07-Human-smpl')
subjects_notr = [41,44,46,48,49,51,53,55,57,58,60,62]
subjects_full = [42,47,50,52,54,56,59,61]

sessions_notr   = exp_smpl.get_sessions(subjects_notr,'all')
sessions_notr_12 = exp_smpl.get_sessions(subjects_notr,[1,2])
sessions_notr34 = exp_smpl.get_sessions(subjects_notr,[3,4])
sessions_notr_1 = exp_smpl.get_sessions(subjects_notr,[1])
sessions_notr_4 = exp_smpl.get_sessions(subjects_notr,[4])

sessions_full = exp_smpl.get_sessions(subjects_full,'all')
sessions_full_12 = exp_smpl.get_sessions(subjects_full,[1,2])
sessions_full34 = exp_smpl.get_sessions(subjects_full,[3,4])
sessions_full_1 = exp_smpl.get_sessions(subjects_full,[1])
sessions_full_4 = exp_smpl.get_sessions(subjects_full,[4])

#-----------------------------------------------------
# Old data
#-----------------------------------------------------

exp_notr = he.experiment('2015-08-13-Human_notr')
subjects_notr_o = [22,23,25,26,28,29,30]
sessions_notr_o_1 = exp_notr.get_sessions(subjects_notr_o,[1])
sessions_notr_o_4 = exp_notr.get_sessions(subjects_notr_o,[4])

exp_full = he.experiment('2015-08-13-Human_full')
sessions_full_o_1 = exp_full.get_sessions('all',[1])
sessions_full_o_4 = exp_full.get_sessions('all',[4])

#-----------------------------------------------------
# Combined data
#-----------------------------------------------------

sessions_notr_c_1 = sessions_notr_1 + sessions_notr_o_1
sessions_notr_c_4 = sessions_notr_4 + sessions_notr_o_4

sessions_full_c_1 = sessions_full_1 + sessions_full_o_1
sessions_full_c_4 = sessions_full_4 + sessions_full_o_4

#-----------------------------------------------------
# fits
#-----------------------------------------------------

fit_lr_notr_c_1 = mf.fit_population(sessions_notr_c_1, agent_lr)
fit_lr_notr_c_4 = mf.fit_population(sessions_notr_c_4, agent_lr)

fit_lr_full_c_1 = mf.fit_population(sessions_full_c_1, agent_lr)
fit_lr_full_c_4 = mf.fit_population(sessions_full_c_4, agent_lr)

sub_order_notr = rp.pop_scatter_plot(fit_lr_notr_c_4, col = 'sID', 
                 sort_param = 'trCR_x_out', title = 'NOTR 4', fig_no = 2)
rp.pop_scatter_plot(fit_lr_notr_c_1, col = 'sID', 
                 subjects = sub_order_notr, title = 'NOTR 1', fig_no = 1)

sub_order_full = rp.pop_scatter_plot(fit_lr_full_c_4, col = 'sID', 
                 sort_param = 'trCR_x_out', title = 'Full 4', fig_no = 4)
rp.pop_scatter_plot(fit_lr_full_c_1, col = 'sID', 
                 subjects = sub_order_full, title = 'Full 1', fig_no = 3)
