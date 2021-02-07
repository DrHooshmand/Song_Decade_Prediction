# Extracting hd5 files into a single file and transforming them to single Numpy array


## Aggregating

I transformed all these files to python3. First, to add all hd5 to a single file, modify the maindir variable inside the main part of the "create_aggregate_file.py" to the data folders of hd5 and specify the output file name. After about 45 mins, you get the single aggregate hd5.

## Extracting data to a Numpy array

Now, read in the output file generated in the last section in "hdf5_getters_mod.py" file. Then, you'll get a numpy array in "result2" array. In "result_nonz" array, I removed any data that didn't specify the song year. 

The features in each column of data is as follows: <br />
1 get_analysis_sample_rate <br />
2 get_artist_7digitalid <br />
3 get_artist_familiarity <br />
4 get_artist_hotttnesss <br />
5 get_artist_latitude <br />
6 get_artist_longitude <br />
7 get_artist_playmeid <br />
8 get_artist_terms_freq <br />
9 get_artist_terms_weight <br />
10 get_bars_confidence <br />
11 get_bars_start <br />
12 get_beats_confidence <br />
13 get_beats_start <br />
14 get_danceability <br />
15 get_duration <br />
16 get_end_of_fade_in <br />
17 get_energy <br />
18 get_key <br />
19 get_loudness <br />
20 get_mode <br />
21 get_mode_confidence <br />
22 get_release <br />
23 get_release_7digitalid <br />
24 get_sections_confidence <br />
25 get_sections_start <br />
26 get_segments_confidence <br />
27 get_segments_loudness_max <br />
28 get_segments_loudness_max_time <br />
29 get_segments_loudness_start <br />
30 get_segments_pitches <br />
31 get_segments_start <br />
32 get_segments_timbre <br />
33 get_song_hotttnesss <br />
34 get_song_id <br />
35 get_start_of_fade_out <br />
36 get_tatums_confidence <br />
37 get_tatums_start <br />
38 get_tempo <br />
39 get_time_signature <br />
40 get_time_signature_confidence <br />
41 get_track_7digitalid <br />
42 get_year <br />
