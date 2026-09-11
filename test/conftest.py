import os

# Prevent DVCLive tests from attempting remote DVC Studio communication.
os.environ.setdefault('DVC_STUDIO_OFFLINE', 'true')
