/* Network clients for native programs: TCP connections (the database and cache protocols are written in Apsi on top),
   TLS through the system's own library (OpenSSL's libssl, loaded when first used, on Linux and macOS), and one HTTP(S)
   request function (WinHTTP on Windows: its TLS, certificates and proxy settings are the system's). Nothing is linked
   in and nothing has to be installed. Included by apsi_tensor.c after apsi_lib.c (sockets, sock_t, lib_str). */

#ifdef _WIN32
#include <winhttp.h>
#else
#include <strings.h>
#endif

static int net_ready = 0;
static void net_init(void) {
    if (net_ready) return;
#ifdef _WIN32
    WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    net_ready = 1;
}
static AV net_err(const char *kind, const char *msg) { AV p[3] = {a_str("error"), a_str(kind), a_str(msg)}; return a_tuple(3, p); }

/* ---- TLS through OpenSSL (POSIX) */
#ifndef _WIN32
static struct {
    int state; void *ssl, *crypto; void *ctx;
    int (*init_ssl)(uint64_t, const void *); const void *(*client_method)(void);
    void *(*ctx_new)(const void *); int (*ctx_default_paths)(void *); int (*ctx_load)(void *, const char *, const char *);
    void (*ctx_verify)(void *, int, void *); void *(*ssl_new)(void *); int (*set_fd)(void *, int);
    long (*ctrl)(void *, int, long, void *); int (*set1_host)(void *, const char *); int (*connect)(void *);
    int (*read)(void *, void *, int); int (*write)(void *, const void *, int); int (*shutdown)(void *); void (*free_)(void *);
    long (*verify_result)(const void *); int (*get_error)(const void *, int);
    unsigned long (*err_get)(void); void (*err_string)(unsigned long, char *, size_t);
    const char *(*verify_string)(long);
} ossl;
static int ossl_load(void) {
    if (ossl.state) return ossl.state > 0;
    ossl.state = -1;
    const char *ssl_names[] = {getenv("APSI_LIBSSL"), "libssl.so.3", "libssl.so.1.1", "libssl.so", "libssl.3.dylib", "/usr/lib/libssl.dylib", NULL};
    const char *cr_names[] = {getenv("APSI_LIBCRYPTO"), "libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so", "libcrypto.3.dylib", "/usr/lib/libcrypto.dylib", NULL};
    for (int i = 0; i < 6 && !ossl.ssl; i++) if (ssl_names[i] && *ssl_names[i]) ossl.ssl = dlopen(ssl_names[i], RTLD_NOW | RTLD_GLOBAL);
    for (int i = 0; i < 6 && !ossl.crypto; i++) if (cr_names[i] && *cr_names[i]) ossl.crypto = dlopen(cr_names[i], RTLD_NOW | RTLD_GLOBAL);
    if (!ossl.ssl) return 0;
    #define OS(f, n) *(void **)&ossl.f = dlsym(ossl.ssl, n)
    #define OC(f, n) *(void **)&ossl.f = ossl.crypto ? dlsym(ossl.crypto, n) : NULL
    OS(init_ssl, "OPENSSL_init_ssl"); OS(client_method, "TLS_client_method"); OS(ctx_new, "SSL_CTX_new");
    OS(ctx_default_paths, "SSL_CTX_set_default_verify_paths"); OS(ctx_load, "SSL_CTX_load_verify_locations");
    OS(ctx_verify, "SSL_CTX_set_verify"); OS(ssl_new, "SSL_new"); OS(set_fd, "SSL_set_fd"); OS(ctrl, "SSL_ctrl");
    OS(set1_host, "SSL_set1_host"); OS(connect, "SSL_connect"); OS(read, "SSL_read"); OS(write, "SSL_write");
    OS(shutdown, "SSL_shutdown"); OS(free_, "SSL_free"); OS(verify_result, "SSL_get_verify_result"); OS(get_error, "SSL_get_error");
    OC(err_get, "ERR_get_error"); OC(err_string, "ERR_error_string_n"); OC(verify_string, "X509_verify_cert_error_string");
    #undef OS
    #undef OC
    if (!ossl.client_method || !ossl.ctx_new || !ossl.ssl_new || !ossl.set_fd || !ossl.connect || !ossl.read || !ossl.write || !ossl.ctrl) return 0;
    if (ossl.init_ssl) ossl.init_ssl(0, NULL);
    ossl.ctx = ossl.ctx_new(ossl.client_method());
    if (!ossl.ctx) return 0;
    const char *ca = getenv("APSI_CA_FILE"); if (!ca || !*ca) ca = getenv("SSL_CERT_FILE");
    if (ca && *ca && ossl.ctx_load) ossl.ctx_load(ossl.ctx, ca, NULL);
    if (ossl.ctx_default_paths) ossl.ctx_default_paths(ossl.ctx);
    if (ossl.ctx_verify) ossl.ctx_verify(ossl.ctx, 1, NULL);              /* SSL_VERIFY_PEER */
    ossl.state = 1; return 1;
}
#endif

/* ---- connections: a table of open sockets (and their TLS sessions), handed to Apsi as small integers */
typedef struct { sock_t fd; void *tls; int used; double timeout; } NetConn;
static NetConn *net_tab; static int net_n;
static int net_slot(void) {
    for (int i = 0; i < net_n; i++) if (!net_tab[i].used) return i;
    net_tab = (NetConn *)realloc(net_tab, sizeof(NetConn) * (size_t)(net_n + 16)); memset(net_tab + net_n, 0, sizeof(NetConn) * 16);
    net_n += 16; return net_n - 16;
}
static NetConn *net_get(AV h) { if (h.t != T_INT || h.i < 1 || h.i > net_n || !net_tab[h.i - 1].used) a_error("OSError", "this connection is closed"); return &net_tab[h.i - 1]; }

static void net_set_timeout(sock_t fd, double t) {
#ifdef _WIN32
    DWORD ms = t > 0 ? (DWORD)(t * 1000) : 0; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms);
#else
    struct timeval tv; tv.tv_sec = (time_t)t; tv.tv_usec = (suseconds_t)((t - (double)(time_t)t) * 1e6);
    if (t <= 0) { tv.tv_sec = 0; tv.tv_usec = 0; }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

static AV net_connect(const char *host, int64_t port, double timeout) {
    net_init();
    struct addrinfo hints, *ai = NULL, *p; memset(&hints, 0, sizeof hints); hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    char ps[16]; snprintf(ps, sizeof ps, "%lld", (long long)port);
    int rc = getaddrinfo(host, ps, &hints, &ai);
    if (rc || !ai) { char m[300]; snprintf(m, sizeof m, "cannot resolve the host name '%s'", host); return net_err("ConnectionError", m); }
    sock_t fd = SOCK_BAD; int refused = 0, timed = 0;
    for (p = ai; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd == SOCK_BAD) continue;
#ifdef _WIN32
        u_long nb = 1; ioctlsocket(fd, FIONBIO, &nb);
#else
        int fl = fcntl(fd, F_GETFL, 0); fcntl(fd, F_SETFL, fl | O_NONBLOCK);
#endif
        int c = connect(fd, p->ai_addr, (int)p->ai_addrlen);
        int ok = c == 0;
        if (!ok) {
            int e = sock_err();
#ifdef _WIN32
            int pending = e == WSAEWOULDBLOCK;
#else
            int pending = e == EINPROGRESS;
#endif
            if (pending) {
                struct pollfd pf = {fd, POLLOUT, 0};
                int w = sock_poll(&pf, 1, timeout > 0 ? (int)(timeout * 1000) : -1);
                if (w > 0) { int soerr = 0; socklen_t sl = sizeof soerr; getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *)&soerr, &sl); ok = soerr == 0; if (!ok) refused = 1; }
                else if (w == 0) timed = 1;
            } else refused = 1;
        }
        if (ok) {
#ifdef _WIN32
            u_long b = 0; ioctlsocket(fd, FIONBIO, &b);
#else
            fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
#endif
            int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
            break;
        }
        sock_close(fd); fd = SOCK_BAD;
    }
    freeaddrinfo(ai);
    if (fd == SOCK_BAD) {
        char m[300];
        if (timed) { snprintf(m, sizeof m, "timed out connecting to %s:%lld", host, (long long)port); return net_err("TimeoutError", m); }
        snprintf(m, sizeof m, "[Errno 111] Connection refused (%s:%lld)", host, (long long)port); (void)refused;
        return net_err("ConnectionRefusedError", m);
    }
    net_set_timeout(fd, timeout);
    int s = net_slot(); net_tab[s].fd = fd; net_tab[s].tls = NULL; net_tab[s].used = 1; net_tab[s].timeout = timeout;
    return a_int(s + 1);
}

static AV net_tls(NetConn *c, const char *host) {
#ifdef _WIN32
    (void)c; (void)host;
    return net_err("OSError", "TLS on a raw connection is not available in native programs on Windows yet (HTTPS works: it goes through WinHTTP)");
#else
    if (!ossl_load()) return net_err("OSError", "TLS needs OpenSSL's libssl (install the openssl package; APSI_LIBSSL can name the library file)");
    void *s = ossl.ssl_new(ossl.ctx); if (!s) return net_err("OSError", "TLS: cannot start a session");
    ossl.set_fd(s, (int)c->fd);
    ossl.ctrl(s, 55, 0, (void *)host);                         /* SSL_set_tlsext_host_name: SNI */
    if (ossl.set1_host) ossl.set1_host(s, host);              /* the certificate must name this host */
    if (ossl.connect(s) != 1) {
        char m[512] = "TLS handshake failed";
        long vr = ossl.verify_result ? ossl.verify_result(s) : 0;
        if (vr && ossl.verify_string) snprintf(m, sizeof m, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: %s", ossl.verify_string(vr));
        else if (ossl.err_get && ossl.err_string) { unsigned long e = ossl.err_get(); if (e) { char t[256]; ossl.err_string(e, t, sizeof t); snprintf(m, sizeof m, "TLS handshake failed: %s", t); } }
        ossl.free_(s); return net_err("SSLError", m);
    }
    c->tls = s; return A_NONE;
#endif
}

static int64_t net_send_all(NetConn *c, const char *p, int64_t n) {
    int64_t done = 0;
    while (done < n) {
        int64_t k;
#ifndef _WIN32
        if (c->tls) k = ossl.write(c->tls, p + done, (int)(n - done > 1 << 30 ? 1 << 30 : n - done)); else
#endif
        k = send(c->fd, p + done, (int)(n - done > 1 << 30 ? 1 << 30 : n - done), 0);
        if (k <= 0) return -1;
        done += k;
    }
    return done;
}
static int64_t net_recv(NetConn *c, char *buf, int64_t cap) {
#ifndef _WIN32
    if (c->tls) { int k = ossl.read(c->tls, buf, (int)cap); if (k <= 0) { int e = ossl.get_error ? ossl.get_error(c->tls, k) : 0; return e == 6 || e == 0 ? 0 : -1; } return k; }
#endif
    return recv(c->fd, buf, (int)cap, 0);
}
static void net_close(NetConn *c) {
#ifndef _WIN32
    if (c->tls) { ossl.shutdown(c->tls); ossl.free_(c->tls); }
#endif
    sock_close(c->fd); c->used = 0; c->tls = NULL;
}
static int net_timed_out(void) {
#ifdef _WIN32
    return WSAGetLastError() == WSAETIMEDOUT;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

/* ---- HTTP(S): one request, the whole answer: (status, reason, headers as [[name, value]], body bytes) */
typedef struct { char scheme[8]; char host[256]; int port; char path[4096]; } NetURL;
static int net_parse_url(const char *u, NetURL *o) {
    memset(o, 0, sizeof *o);
    const char *p = strstr(u, "://"); if (!p) return 0;
    size_t sl = (size_t)(p - u); if (sl >= sizeof o->scheme) return 0;
    for (size_t i = 0; i < sl; i++) o->scheme[i] = (char)(u[i] >= 'A' && u[i] <= 'Z' ? u[i] + 32 : u[i]);
    if (strcmp(o->scheme, "http") && strcmp(o->scheme, "https")) return 0;
    p += 3; const char *hs = p; while (*p && *p != '/' && *p != '?' && *p != '#') p++;
    const char *at = NULL; for (const char *q = hs; q < p; q++) if (*q == '@') at = q;
    if (at) hs = at + 1;
    const char *he = p, *colon = NULL;
    if (*hs == '[') { const char *rb = memchr(hs, ']', (size_t)(he - hs)); if (!rb) return 0; if (rb + 1 < he && rb[1] == ':') colon = rb + 1; snprintf(o->host, sizeof o->host, "%.*s", (int)(rb - hs - 1), hs + 1); }
    else { for (const char *q = hs; q < he; q++) if (*q == ':') colon = q; snprintf(o->host, sizeof o->host, "%.*s", (int)((colon ? colon : he) - hs), hs); }
    o->port = colon ? atoi(colon + 1) : (!strcmp(o->scheme, "https") ? 443 : 80);
    const char *path = *p == '/' || *p == '?' ? p : "/";
    const char *hash = strchr(path, '#');
    snprintf(o->path, sizeof o->path, "%s%.*s", *path == '?' ? "/" : "", (int)(hash ? hash - path : (long)strlen(path)), path);
    return o->host[0] != 0;
}

#ifdef _WIN32
static wchar_t *net_w(const char *s) { int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0); wchar_t *w = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)(n + 1)); MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n); return w; }
static char *net_u8(const wchar_t *w, int len) { int n = WideCharToMultiByte(CP_UTF8, 0, w, len, NULL, 0, NULL, NULL); char *s = (char *)malloc((size_t)n + 1); WideCharToMultiByte(CP_UTF8, 0, w, len, s, n, NULL, NULL); s[n] = 0; return s; }
static AV net_winhttp(const char *method, const char *url, AV headers, AV body, double timeout) {
    NetURL u; if (!net_parse_url(url, &u)) return net_err("ValueError", "not an http:// or https:// URL");
    char m[512];
    HINTERNET ses = WinHttpOpen(L"apsi", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!ses) ses = WinHttpOpen(L"apsi", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!ses) return net_err("OSError", "WinHTTP is not available");
    int ms = timeout > 0 ? (int)(timeout * 1000) : 0;
    WinHttpSetTimeouts(ses, ms, ms, ms, ms);
    wchar_t *wh = net_w(u.host), *wp = net_w(u.path), *wm = net_w(method);
    HINTERNET con = WinHttpConnect(ses, wh, (INTERNET_PORT)u.port, 0);
    HINTERNET req = con ? WinHttpOpenRequest(con, wm, wp, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, !strcmp(u.scheme, "https") ? WINHTTP_FLAG_SECURE : 0) : NULL;
    free(wh); free(wp); free(wm);
    AV result = A_NONE;
    if (!req) { snprintf(m, sizeof m, "cannot open a request to %s (error %lu)", u.host, (unsigned long)GetLastError()); result = net_err("ConnectionError", m); goto done; }
    DWORD off = WINHTTP_DISABLE_REDIRECTS; WinHttpSetOption(req, WINHTTP_OPTION_DISABLE_FEATURE, &off, sizeof off);
    if (a_is(headers, O_LIST)) for (int64_t i = 0; i < ((AList *)headers.o)->len; i++) {
        AV pr = ((AList *)headers.o)->v[i]; AStr *k = (AStr *)((AList *)pr.o)->v[0].o, *v = (AStr *)((AList *)pr.o)->v[1].o;
        size_t L = (size_t)(k->len + v->len + 4); char *line = (char *)malloc(L + 1); snprintf(line, L + 1, "%s: %s", k->s, v->s);
        wchar_t *wl = net_w(line); WinHttpAddRequestHeaders(req, wl, (DWORD)-1L, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE); free(wl); free(line);
    }
    const char *bp = body.t == T_OBJ ? ((AStr *)body.o)->s : NULL; DWORD bl = body.t == T_OBJ ? (DWORD)((AStr *)body.o)->len : 0;
    if (!WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0, (LPVOID)bp, bl, bl, 0) || !WinHttpReceiveResponse(req, NULL)) {
        DWORD e = GetLastError();
        if (e == ERROR_WINHTTP_TIMEOUT) { snprintf(m, sizeof m, "timed out (%s)", u.host); result = net_err("TimeoutError", m); }
        else if (e == ERROR_WINHTTP_CANNOT_CONNECT) { snprintf(m, sizeof m, "[WinError 12029] cannot connect to %s:%d", u.host, u.port); result = net_err("ConnectionRefusedError", m); }
        else if (e == ERROR_WINHTTP_NAME_NOT_RESOLVED) { snprintf(m, sizeof m, "cannot resolve the host name '%s'", u.host); result = net_err("ConnectionError", m); }
        else if (e == ERROR_WINHTTP_SECURE_FAILURE) { snprintf(m, sizeof m, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (%s)", u.host); result = net_err("SSLError", m); }
        else { snprintf(m, sizeof m, "HTTP request to %s failed (WinHTTP error %lu)", u.host, (unsigned long)e); result = net_err("ConnectionError", m); }
        goto done;
    }
    {
        DWORD code = 0, sz = sizeof code;
        WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &sz, WINHTTP_NO_HEADER_INDEX);
        DWORD hl = 0; WinHttpQueryHeaders(req, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &hl, WINHTTP_NO_HEADER_INDEX);
        wchar_t *wraw = (wchar_t *)malloc(hl + 2); WinHttpQueryHeaders(req, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, wraw, &hl, WINHTTP_NO_HEADER_INDEX);
        char *raw = net_u8(wraw, (int)(hl / sizeof(wchar_t))); free(wraw);
        int sp = a_ntmp; AV hs = a_list(0, NULL); A_PUSH(hs); AV reason = a_str("");
        char *line = raw, *nl; int first = 1;
        while (line && *line) {
            nl = strstr(line, "\r\n"); if (nl) *nl = 0;
            if (first) { const char *r = strchr(line, ' '); if (r) r = strchr(r + 1, ' '); reason = a_str(r ? r + 1 : ""); A_PUSH(reason); first = 0; }
            else { char *c = strchr(line, ':'); if (c) { *c = 0; char *v = c + 1; while (*v == ' ') v++; AV pr[2] = {a_str(line), a_str(v)}; a_list_append((AList *)hs.o, a_list(2, pr)); } }
            line = nl ? nl + 2 : NULL;
        }
        free(raw);
        FrBuf b = {0}; DWORD avail = 0;
        while (WinHttpQueryDataAvailable(req, &avail) && avail) {
            char *chunk = (char *)malloc(avail); DWORD got = 0;
            if (!WinHttpReadData(req, chunk, avail, &got)) { free(chunk); break; }
            fr_put(&b, chunk, (size_t)((int64_t)got)); free(chunk);
        }
        AV bv = a_bytes(b.p ? b.p : "", b.len); free(b.p); A_PUSH(bv);
        AV parts[4] = {a_int((int64_t)code), reason, hs, bv}; result = a_tuple(4, parts); a_ntmp = sp;
    }
done:
    if (req) WinHttpCloseHandle(req); if (con) WinHttpCloseHandle(con); WinHttpCloseHandle(ses);
    return result;
}
#endif

/* POSIX: the request on a socket (through an HTTP proxy from the environment when one is set), TLS by OpenSSL */
#ifndef _WIN32
static int net_no_proxy(const char *host) {
    const char *np = getenv("NO_PROXY"); if (!np) np = getenv("no_proxy"); if (!np || !*np) return 0;
    char buf[1024]; snprintf(buf, sizeof buf, "%s", np);
    for (char *t = strtok(buf, ", "); t; t = strtok(NULL, ", ")) {
        if (!strcmp(t, "*")) return 1;
        const char *d = *t == '.' ? t + 1 : t; size_t hl = strlen(host), dl = strlen(d);
        if (hl >= dl && !strcasecmp(host + hl - dl, d) && (hl == dl || host[hl - dl - 1] == '.')) return 1;
    }
    return 0;
}
static AV net_posix_http(const char *method, const char *url, AV headers, AV body, double timeout) {
    NetURL u; if (!net_parse_url(url, &u)) return net_err("ValueError", "not an http:// or https:// URL");
    int https = !strcmp(u.scheme, "https");
    const char *pe = NULL;
    if (!net_no_proxy(u.host) && strcmp(u.host, "localhost") && strcmp(u.host, "127.0.0.1") && strcmp(u.host, "::1")) {
        pe = https ? (getenv("HTTPS_PROXY") ? getenv("HTTPS_PROXY") : getenv("https_proxy")) : (getenv("HTTP_PROXY") ? getenv("HTTP_PROXY") : getenv("http_proxy"));
        if (pe && !*pe) pe = NULL;
    }
    NetURL pu; if (pe && !net_parse_url(pe, &pu)) { char t[600]; snprintf(t, sizeof t, "http://%s", pe); if (!net_parse_url(t, &pu)) pe = NULL; }
    AV h = pe ? net_connect(pu.host, pu.port, timeout) : net_connect(u.host, u.port, timeout);
    if (h.t != T_INT) return h;
    NetConn *c = &net_tab[h.i - 1]; char m[600]; FrBuf rq = {0};
    if (pe && https) {                                         /* a tunnel through the proxy */
        char t[700]; snprintf(t, sizeof t, "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n\r\n", u.host, u.port, u.host, u.port);
        if (net_send_all(c, t, (int64_t)strlen(t)) < 0) { net_close(c); return net_err("ConnectionError", "the proxy closed the connection"); }
        FrBuf r = {0}; char ch;
        while (r.len < 8192 && net_recv(c, &ch, 1) == 1) { fr_put(&r, &ch, (size_t)(1)); if (r.len >= 4 && !memcmp(r.p + r.len - 4, "\r\n\r\n", 4)) break; }
        int ok = r.len > 12 && r.p[9] == '2'; free(r.p);
        if (!ok) { net_close(c); return net_err("ConnectionError", "the proxy refused the tunnel"); }
    }
    if (https) { AV t = net_tls(c, u.host); if (t.t != T_NONE) { net_close(c); return t; } }
    if (pe && !https) snprintf(m, sizeof m, "%s %s HTTP/1.1\r\n", method, url); else snprintf(m, sizeof m, "%s %s HTTP/1.1\r\n", method, u.path);
    fr_put(&rq, m, strlen(m));
    if ((u.port == 80 && !https) || (u.port == 443 && https)) snprintf(m, sizeof m, "Host: %s\r\n", u.host); else snprintf(m, sizeof m, "Host: %s:%d\r\n", u.host, u.port);
    fr_put(&rq, m, strlen(m)); fr_put(&rq, "Accept-Encoding: identity\r\nConnection: close\r\n", strlen("Accept-Encoding: identity\r\nConnection: close\r\n"));
    int has_len = 0;
    if (a_is(headers, O_LIST)) for (int64_t i = 0; i < ((AList *)headers.o)->len; i++) {
        AV pr = ((AList *)headers.o)->v[i]; AStr *k = (AStr *)((AList *)pr.o)->v[0].o, *v = (AStr *)((AList *)pr.o)->v[1].o;
        if (!strcasecmp(k->s, "content-length")) has_len = 1;
        fr_put(&rq, k->s, (size_t)(k->len)); fr_put(&rq, ": ", strlen(": ")); fr_put(&rq, v->s, (size_t)(v->len)); fr_put(&rq, "\r\n", strlen("\r\n"));
    }
    int64_t bl = body.t == T_OBJ ? ((AStr *)body.o)->len : 0;
    if (!has_len && (bl || strcmp(method, "GET"))) { snprintf(m, sizeof m, "Content-Length: %lld\r\n", (long long)bl); fr_put(&rq, m, strlen(m)); }
    fr_put(&rq, "\r\n", strlen("\r\n")); if (bl) fr_put(&rq, ((AStr *)body.o)->s, (size_t)(bl));
    int64_t sent = net_send_all(c, rq.p, rq.len); free(rq.p);
    if (sent < 0) { net_close(c); return net_err("ConnectionError", "the server closed the connection"); }
    FrBuf r = {0}; char buf[65536]; int64_t k;
    while ((k = net_recv(c, buf, sizeof buf)) > 0) fr_put(&r, buf, (size_t)(k));
    int to = k < 0 && net_timed_out();
    net_close(c);
    if (r.p) r.p[r.len] = 0;
    if (to) { free(r.p); snprintf(m, sizeof m, "timed out reading from %s", u.host); return net_err("TimeoutError", m); }
    char *he = r.p ? strstr(r.p, "\r\n\r\n") : NULL;
    if (!r.p || r.len < 12 || memcmp(r.p, "HTTP/", 5) || !he) { free(r.p); return net_err("ConnectionError", "the server's answer is not HTTP"); }
    *he = 0;
    int code = atoi(strchr(r.p, ' ') ? strchr(r.p, ' ') + 1 : "0");
    int sp = a_ntmp; AV hs = a_list(0, NULL); A_PUSH(hs); AV reason = a_str(""); A_PUSH(reason);
    { const char *sp1 = strchr(r.p, ' '); const char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL; const char *eol = strstr(r.p, "\r\n"); if (sp2 && (!eol || sp2 < eol)) reason = a_strn(sp2 + 1, (eol ? eol : sp2 + 1 + strlen(sp2 + 1)) - sp2 - 1); a_tmp[a_ntmp - 1] = reason; }
    int chunked = 0; int64_t clen = -1;
    for (char *line = strstr(r.p, "\r\n"); line; ) {
        line += 2; char *nl = strstr(line, "\r\n"); if (nl) *nl = 0;
        char *colon = strchr(line, ':');
        if (colon) {
            *colon = 0; char *v = colon + 1; while (*v == ' ' || *v == '\t') v++;
            if (!strcasecmp(line, "transfer-encoding") && strstr(v, "chunked")) chunked = 1;
            if (!strcasecmp(line, "content-length")) clen = atoll(v);
            AV pr[2] = {a_str(line), a_str(v)}; a_list_append((AList *)hs.o, a_list(2, pr));
        }
        line = nl;
    }
    char *bs = he + 4; int64_t bn = (int64_t)r.len - (bs - r.p);
    FrBuf out = {0};
    if (chunked) {
        char *q = bs, *e = bs + bn;
        while (q < e) {
            int64_t n = strtoll(q, NULL, 16); char *nl = strstr(q, "\r\n"); if (!nl || n <= 0) break;
            q = nl + 2; if (q + n > e) n = e - q; fr_put(&out, q, (size_t)(n)); q += n + 2;
        }
    } else fr_put(&out, bs, (size_t)(clen >= 0 && clen < bn ? clen : bn));
    AV bv = a_bytes(out.p ? out.p : "", out.len); free(out.p); free(r.p); A_PUSH(bv);
    AV parts[4] = {a_int(code), reason, hs, bv}; AV res = a_tuple(4, parts); a_ntmp = sp; return res;
}
#endif

BI(lb_net) {
    (void)self; (void)kw; const char *op = lib_str(a[0], "net")->s;
    if (!strcmp(op, "connect")) return net_connect(lib_str(a[1], "net")->s, a[2].i, n > 3 ? a_c_f64(a[3]) : 0);
    if (!strcmp(op, "tls")) return net_tls(net_get(a[1]), lib_str(a[2], "net")->s);
    if (!strcmp(op, "send")) {
        NetConn *c = net_get(a[1]); AStr *d = lib_str(a[2], "net");
        if (net_send_all(c, d->s, d->len) < 0) return net_err("ConnectionResetError", "the connection was closed by the other side");
        return a_int(d->len);
    }
    if (!strcmp(op, "recv")) {
        NetConn *c = net_get(a[1]); int64_t cap = n > 2 ? a[2].i : 65536; if (cap < 1) cap = 1; if (cap > 1 << 24) cap = 1 << 24;
        char *buf = (char *)malloc((size_t)cap); int64_t k = net_recv(c, buf, cap);
        if (k < 0) { free(buf); if (net_timed_out()) return net_err("TimeoutError", "timed out"); return net_err("ConnectionResetError", "the connection was reset"); }
        AV r = a_bytes(buf, k); free(buf); return r;
    }
    if (!strcmp(op, "timeout")) { NetConn *c = net_get(a[1]); c->timeout = a_c_f64(a[2]); net_set_timeout(c->fd, c->timeout); return A_NONE; }
    if (!strcmp(op, "close")) { if (a[1].t == T_INT && a[1].i >= 1 && a[1].i <= net_n && net_tab[a[1].i - 1].used) net_close(&net_tab[a[1].i - 1]); return A_NONE; }
    if (!strcmp(op, "request")) {
        const char *method = lib_str(a[1], "net")->s, *url = lib_str(a[2], "net")->s; double t = n > 5 ? a_c_f64(a[5]) : 30;
#ifdef _WIN32
        net_init(); return net_winhttp(method, url, n > 3 ? a[3] : A_NONE, n > 4 ? a[4] : A_NONE, t);
#else
        return net_posix_http(method, url, n > 3 ? a[3] : A_NONE, n > 4 ? a[4] : A_NONE, t);
#endif
    }
    a_error("ValueError", "unknown network operation '%s'", op);
}
