% List of open inputs
nrun = 1; % enter the number of runs here
jobfile = {'/home/local/AIUGA/clm121/Documents/deepnet/deepnet/rbm_piglets/spm_normalize_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
