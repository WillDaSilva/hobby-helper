var http = require('http');
var WebSocketServer = require('ws').Server; //provides web sockets
var ecStatic = require('ecstatic');  //provides static file server service
var url = require('url');

//static file server
var server = http.createServer(ecStatic({root: __dirname + '/www'}));

var wss = new WebSocketServer({server: server});
var redditUsernamePattern = /[\w-]{20}/;

wss.on('connection', function(ws) { //function runs when a new connection is opened
	console.log('New client connected');
	
	ws.on('message', function(data){
		if(data){
			var parsedData = JSON.parse(data);
			
			if(parsedData.type == 'profileURL'){
				urlObj = url.parse(parsedData.data); //parse request type
				if(urlObj.hostname === 'www.reddit.com'){
					if(urlObj.pathname.startsWith('/u/')){
						redditUsername = urlObj.pathname.substring(3);
					}
					else if(urlObj.pathname.startsWith('/user/')){
						redditUsername = urlObj.pathname.substring(6);
					}
					if(redditUsernamePattern.test(redditUsername)){
						var request = http.get('https://www.reddit.com/user/' + redditUsername + '/.json', function(response){
							var userObj = JSON.stringify(response);
							if(!userObj.error){
								//Neural net processing goes here
								//websocket return message goes here
							}
						});
					}
					else{
						ws.send(JSON.stringify({type : error, error : 'Not a valid reddit username'}));
					}
				}
			}
			
		}
	});
});

var port = 3000;
server.listen(port); 
console.log('Server started on port ' + port);
