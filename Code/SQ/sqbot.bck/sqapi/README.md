# SQAPI

A submodule to facilitate interactions with the SQ+ API.

The `SQAPI` module is used to handle the interactions with the API. It takes care of authentication, and simplifies the creation of API queries. For example:
```python
sqapi = SQAPI(host=<HOST>,api_key=<API_KEY>, verbosity=2)  # instantiate the sqapi module

r=sqapi.get(<ENDPOINT>)                 # define a get request using a specific endpoint
  r.filter(<NAME>,<OPERATORE>,<VALUE>)  # define a filter to compare a property with a value using an operator
  r.template(<TEMPLATE>)                # format the output of the request using an inbuilt HTML template
  html = r.execute().text               # perform the request & return result as text (eg: for html)
  data = r.execute().json()             # perform the request & return result as JSON dict (don't set template)
```

The `sqapi` module helps to build the `HTTP` requests that are sent to the `API`. These are essentially just URLS in most cases (`GET` requests). Setting `verbosity=2` on the `sqapi` module will print the `HTTP` requests that are being made.