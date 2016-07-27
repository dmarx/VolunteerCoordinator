Solves volunteer coordination problems using linear programming.

Target feature set:

* Assigning volunteers to duty shifts, subject to preference constraints possibly including:
    * Shift preferences
    * Reducing number of shifts on same day for multi-day events
    * Adding preference to sequential shifts when there are multiple shifts on the same day. (I think that makes sense?)
    * Pairing people who want to work together
    * Spreading out hours across as many people as possible as uniformly as possible (i.e. sharing the load as much as possible)

* Assigning volunteers to housing, subject to preference constraints possibly including:
    * Gender preference
    * Pairing people who are travelling together or would prefer to be housed together
    * Host's available space
    
* Identifying areas of weakness, including:
    * Identifying which roles need to be filled when an optimal duty assignment solution does not exist
    * Identifying housing needs when an optimal housing solution does not exist