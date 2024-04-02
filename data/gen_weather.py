import urllib
import json
import time
from datetime import datetime, timedelta

zip,city = "90001", "Los Angeles"
client_secret = "xxxxxxxxxxx" # Need to get it on the official website. 
start_date = datetime(2022, 9, 1)
days_to_increment = 1
current_date = start_date

while current_date<=datetime(2023, 7, 6):
	time.sleep(10)
	formatted_date = current_date.strftime("%m/%d/%Y")
	request = urllib.request.urlopen("https://api.aerisapi.com/observations/summary/"+zip+"?from="+formatted_date+"&format=json&filter=mesonet,hasPrecip&limit=22&fields=id,loc,periods.summary.dateTimeISO,periods.summary.temp&client_id=4JEl5pr4w7Kkizinq43E2&client_secret="+client_secret)
	response = request.read()
	json_info = json.loads(response)
	if json_info["success"]:
		file_path = "./aerisapi/"+zip+"-"+current_date.strftime("%Y-%m-%d")+".json"
		with open(file_path, "w") as json_file:
			json.dump(json_info, json_file)
		print("write ",file_path)
	else:
		print("An error occurred: %s" % (json_info["error"]["description"]))
		request.close()
	current_date += timedelta(days=days_to_increment)
time.sleep(15)