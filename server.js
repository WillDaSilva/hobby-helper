const https = require('https');
const http = require('http');
const WebSocketServer = require('ws').Server; //provides web sockets
const ecStatic = require('ecstatic');  //provides static file server service
const url = require('url');
const fs = require('fs');

//static file server
let server = http.createServer(ecStatic({root: __dirname + '/client'}));

let wss = new WebSocketServer({server: server});
let redditUsernamePattern = /[\w-]{3,20}/;

let hobbies = JSON.parse(fs.readFileSync('hobbies.json'));
let hobbiesList = [];

let prepRegex = new RegExp(/(?:\n|[^\w\s])+/, 'g');

for(let key of Object.keys(hobbies)){
	for(let value of hobbies[key]){
		hobbiesList.push(value);
	}
}

wss.on('connection', function(ws) { //function runs when a new connection is opened
    console.log('New client connected');

    ws.on('message', function(data){
        if(!data) return;
		console.log('Received data from client: ' + data);
        let parsedData = JSON.parse(data);
        if(parsedData.type == 'profileURL') {
            urlObj = url.parse(parsedData.data); //parse request type
            if(urlObj.hostname === 'www.reddit.com') {
				console.log('Pathname: ' + urlObj.pathname);
                if(urlObj.pathname.startsWith('/u/')) {
                    redditUsername = urlObj.pathname.substring(3);
                }
                else if(urlObj.pathname.startsWith('/user/')) {
                    redditUsername = urlObj.pathname.substring(6);
                }
				console.log('Username: ' + redditUsername);
                if(redditUsernamePattern.test(redditUsername)) {
					console.log('Username valid');
                    let request = https.get('https://www.reddit.com/user/' + redditUsername + '/.json', function(response){
						let body = '';
						
						response.on('data', function(chunk){
							body += chunk;
						});
						
						response.on('end', function(){
							let userObj = JSON.parse(body);
							if(!userObj.error){
								userString = '';
								for(let child of userObj['data']['children']){
									if(!hobbiesList.includes(child['data']['subreddit'])){
										userString += child['data']['body'] + ' ';
									}
								}
								userString = userString.replace(prepRegex, ' ');
								userString = userString.toLowerCase();
								console.log(userString);
							}
						});
                    }).on('error', function(e){
						console.log('Got an error: ', e);
					});
                }
                else {
                    ws.send(JSON.stringify({type : 'error', error : 'Not a valid reddit username'}));
                }
            }
        }
    });
});

const PORT = 12601;
server.listen(PORT, function(error) {
    if (error) {
        console.log(error);
    }
    else {
        console.log(`Server started on port ${PORT}`);
    }
});
