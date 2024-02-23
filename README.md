# MoonBoard Generator

This is a side-project focused on creating a MoonBoard route generator based on existing data. The aim is to build an end-to-end webapp that can connect to a moonboard via bluetooth and display generated routes.

## Dataset

The older MoonBOard API which was previously used to query information on routes via a REST API has been deprecated (as of 20240221). I've resorted to using [this](https://github.com/spookykat/MoonBoard/issues/6#issuecomment-1783515787) dataset dated 20233001.

I've constrained the model to using routes from MoonBoard 2016 as it has the most route data among all the boards and is quite accessible for me. The model should be transferrable to other MoonBoard versions.

## Model

TODO