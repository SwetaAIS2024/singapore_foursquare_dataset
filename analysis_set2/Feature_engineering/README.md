### Feature Engineering notes :
- Before applying the clustering algorithm on the users.
- New features need to be extrated from the original dataset.
- These features should represent the below:
    - Travel pattern
    - Preferred POI 
    - Preferred time to visit the particular POI
- Possible features include:
    - Count of Unique places visited
    - Frequency of the visits
    - Most active hours - to specify the hour bins 
    - Total checkins / user 
    - Average visits for each place (also per day )


### Making sense of the user_activity_trend_encoded.csv file :
- Query : check how the peak_hour_checkins is calculated ? how it is 44 , greater than 39 which is peak_day_cheackins , just check ithe logic , and do not modify anything yet

- Response :
The value for peak_hour_checkins is calculated as the sum of check-ins in the peak 2-hour window (e.g., 22:00-23:00), across all days for that user.
The value for peak_day_checkins is the total check-ins on the peak day of the week (e.g., all Sundays), across all hours and all weeks.

So, for user 48739:

peak_hour_checkins = 44 means: across all days, there were 44 check-ins in the 22:00-23:00 window.
peak_day_checkins = 39 means: across all hours, there were 39 check-ins on all Sundays.
It is possible for the sum of check-ins in a specific 2-hour window (across all days) to be higher than the total check-ins on the most active day of the week (across all hours), especially if the user is very active in that time window on multiple days.