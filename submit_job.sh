# submits a condor job with the right chmod

chmod a+x download_dataset.sub
chmod a+x download_dataset.py
chmod a+x test.sub
chmod a+x test_condor.py

condor_submit download_dataset.sub

