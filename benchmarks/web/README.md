# HTTP benchmark

A **synthetic** benchmark: one route returning a fixed 25-byte JSON object, keep-alive, loopback, the load generator
on the same machine. It measures the fixed cost per request of the server itself (Apsi's `web.server` with its
routing and security headers, and Node.js's bare `http` module for comparison), not the speed of a real application,
where database calls, request parsing, TLS and the handler's own work usually dominate.

```bash
go build -o load .                     # the load generator (Go 1.21+)
apsi run srv.apsi --allow-net &
node node_server.js &                   # the same route in Node.js, port 58092
./load 127.0.0.1:58090 /hello 256 5     # address, path, connections, seconds
./load 127.0.0.1:58092 /hello 256 5
```

`srv2.apsi` adds a blocking route to show that slow handlers do not hold up fast ones.
