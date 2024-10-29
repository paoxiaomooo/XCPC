[TOC]

## author

#### starlight arithmetic library-paoxiaomo,mrsun,yjh

## 编译

### CF模板

```c++
//#pragma GCC optimize(2,"Ofast")
//#pragma GCC optimize(3,"inline")
//#pragma GCC optimize("unroll-loops")
//#pragma GCC target("avx,avx2,fma")
//#pragma GCC target("sse4,popcnt,abm,mmx")
#include<bits/stdc++.h>
using namespace std;
//#include<bits/extc++.h>
//using namespace __gnu_pbds;
//using namespace __gnu_cxx;

//--------------------------Abbreviations--------------------------//

#define x1 __________________
#define y1 ___________________
#define x2 ____________________
#define y2 _____________________
#define endl '\n'
#define uint unsigned int
#define int long long
using ll = long long;
using ld = long double;
using ull = unsigned long long;
//using lll = __int128;
using vi = vector<int>;
template <class T> using vc = vector<T>;
template <class T> using vvc = vector<vc<T>>;
template <class T> using vvvc = vector<vvc<T>>;
using ai2 = array<int, 2>;using ai3 = array<int, 3>;
using ai4 = array<int, 4>;using ai5 = array<int, 5>;
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
template<class Key> using uset = unordered_set<Key, custom_hash>;
template<class Key, class Value> using umap = unordered_map<Key, Value, custom_hash>;

//--------------------------Function--------------------------//

#define sqrt(x) sqrtl(x)
#define log2(x) (63 - __builtin_clzll(x))
#define Ones(x) __builtin_popcountll(x)
//#define max(a, b) ((a) > (b) ? (a) : (b))
//#define min(a, b) ((a) < (b) ? (a) : (b))
ld getld() { string x;cin >> x;return stold(x); }
template<typename T> inline bool chkmax(T& a, T b) { if (a >= b) return 0;a = b;return 1; }
template<typename T> inline bool chkmin(T& a, T b) { if (a <= b) return 0;a = b;return 1; }

//--------------------------Debug--------------------------//

#ifdef LOCAL
#define debug_(x) cerr << #x << " = " << (x) << ' '
#define debug(x) cerr << #x << " = " << (x) << '\n'
#define debugsq(x) cerr << #x << ": ["; for (auto i : x) cerr << i << ' '; cerr << "]\n";
#define debugmp(x) cerr << #x << ": [ "; for (auto [i, j] : x) cerr << '[' << i << "," << j << "] "; cerr << "]\n";
#define debugs(x...) do{cerr << #x << " : "; ERR(x);} while (0)
void ERR() { cerr << endl; } template <class T, class... Ts> void ERR(T arg, Ts ...args) { cerr << arg << ", ";ERR(args...); }
#else
#define debug(...) 'm'
#define debug_(...) 'r'
#define debugsq(...) 's'
#define debugmp(...) 'u'
#define debugs(...) 'n'
#define ERR() 's'
#endif

//--------------------------Constant--------------------------//

const ll inf = 1e18;
const int N = 2e5 + 10;
const int MOD = 1e9 + 7;//998244353
//const __int128 ONE = 1;

//--------------------------Other--------------------------//

int mx[]{ 0,-1,0,1 };
int my[]{ 1,0,-1,0 };
#define YES cout << "YES" << '\n'
#define NO cout << "NO" << '\n'
#define quit cout << "quit" << '\n'; return
template <class... Args> void myin(Args&... args) { ((cin >> args), ...); }
template <class... Args> void myout(const Args&... args) { ((cout << args), ...); }
//ofstream mcout("C:/Users/Mrsuns/Desktop/out.txt");
//ofstream mcout("C:/Users/Mrsuns/Desktop/out.txt");

//--------------------------END--------------------------//


void PreWork() {
    
}


void Solve(int TIME) {
    
    
    
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cout << fixed << setprecision(15);
    
    PreWork();
    
    int T = 1;
    cin >> T;
    
    int TIME = 0;
    while (T--) {
        Solve(++TIME);
    }
    
    return 0;
}

```





## 字符串

### 字典树

```c++
struct Trie {
    static int nxt[N][26];
    inline static int dic[N];//记录字典中的id//记录字典中是否有该串//记录该串在字典中出现次数
    int root = 0, tot = 0;
    Trie() {}
    ~Trie() { clear(); }
    void clear() {
        for (int i = 0;i <= tot;i++) {
            for (int j = 0;j < 26;j++) nxt[i][j] = 0;
        }
        for (int i = 0;i <= tot;i++) dic[i] = 0;
        root = tot = 0;
    }
    int newnode() {
        return ++tot;
    }
    int Insert(int pre, int ch) {
        return nxt[pre][ch] ? nxt[pre][ch] : nxt[pre][ch] = newnode();
    }
    void insert(string s) {
        int now = root;
        for (int i = 1;i < s.size();i++) {
            now = Insert(now, s[i] - 'a');
        }
        dic[now] += 1;
    }
};
int Trie::nxt[N][26];
//别忘记 s = ' ' + s;

struct Trie01 {
    static int nxt[N][2];
    inline static int dic[N];//记录字典中的id//记录字典中是否有该串//记录该串在字典中出现次数
    inline static int pre[N];//记录该前缀在字典中出现次数
    int root = 0, tot = 0;
    Trie01() {}
    ~Trie01() { clear(); }
    void clear() {
        for (int i = 0;i <= tot;i++) nxt[i][0] = nxt[i][1] = 0;
        for (int i = 0;i <= tot;i++) dic[i] = pre[i] = 0;
        root = tot = 0;
    }
    int newnode() {
        return ++tot;
    }
    int Insert(int pre, int ch) {
        return nxt[pre][ch] ? nxt[pre][ch] : nxt[pre][ch] = newnode();
    }
    void insert(int x, int ID) {
        int now = root;
        for (int i = 30;i >= 0;i--) {
            now = Insert(now, x >> i & 1);
            pre[now] += 1;
        }
        dic[now] += 1;
    }
    void erase(int x) {
        int now = root;
        for (int i = 30;i >= 0;i--) {
            now = nxt[now][x >> i & 1];
            if (pre[now]) pre[now] -= 1;
        }
        if (dic[now]) dic[now] -= 1;
    }
    int query(int x) {
        int res = 0;
        int now = root;
        for (int i = 30;i >= 0;i--) {
            int t = x >> i & 1;
            if (nxt[now][t ^ 1] && pre[nxt[now][t ^ 1]] > 0) now = nxt[now][t ^ 1], res += 1ll << i;
            else now = nxt[now][t];
        }
        return res;
    }

};
int Trie01::nxt[N][2];
//别忘记s=' '+s;


struct PTrie01 {
    static int nxt[N][2];
    inline static int root[N];//第i个版本的根节点编号
    inline static int sum[N];
    int tot = 0;
    PTrie01() {}
    ~PTrie01() { clear(); }
    void clear() {
        for (int i = 0;i <= tot;i++) nxt[i][0] = nxt[i][1] = 0;
        for (int i = 0;i <= tot;i++) sum[i] = 0;
        root[0] = tot = 0;
    }
    int newnode() {
        return ++tot;
    }
    void insert(int& now, int pre, int x) {
        now = newnode();sum[now] = sum[pre] + 1;
        int temp = now;
        for (int i = 25;i >= 0;i--) {
            int t = x >> i & 1;
            nxt[temp][0] = nxt[pre][0];
            nxt[temp][1] = nxt[pre][1];
            nxt[temp][t] = newnode();
            temp = nxt[temp][t]; pre = nxt[pre][t];
            sum[temp] = sum[pre] + 1;
        }
    }
    int query(int l, int r, int x) {
        int res = 0;
        for (int i = 25;i >= 0;i--) {
            int t = x >> i & 1;
            if (sum[nxt[r][t ^ 1]] - sum[nxt[l][t ^ 1]] > 0) {
                res += 1ll << i;
                l = nxt[l][t ^ 1], r = nxt[r][t ^ 1];
            }
            else l = nxt[l][t], r = nxt[r][t];
        }
        return res;
    }
};
int PTrie01::nxt[N][2];
//别忘记s=' '+s;
```

### 最小表示法

```cpp
int min_representation()//字符串最小表示法板子
{
	int i=0,j=1,k=0;
	while (i<n&&j<n&&k<n)
	{
		int t=a[(i+k)%n]-a[(j+k)%n];
		if (t==0) k++;
		else 
		{
			if (t>0) i+=k+1;
			else j+=k+1;
			if (i==j) j++; k=0;
		}
	}
	return min(i,j);	
}

int main()
{
	n=r(); rep(i,0,n-1) a[i]=r(),a[i+n]=a[i];
	int t=min_representation();
	rep(i,t,t+n-1) printf("%d ",a[i]);
	return 0;
}
```

### 扩展KMP(Z函数)

```c++
vector<int> Z_function(const string& s) {//
    int n = (int)s.size() - 1;
    vector<int> z(n + 1);z[1] = n;
    for (int i = 2, l = 0, r = 0;i <= n;i++) {
        if (i <= r) z[i] = min(z[i - l + 1], r - i + 1);
        while (i + z[i] <= n && s[1 + z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
    }
    return z;
}
vector<int> Z_function(const string& s, const string& t, vector<int>& z) {//先计算出s的z函数
    int n = (int)s.size() - 1;
    int m = (int)t.size() - 1;
    vector<int> p(m + 1);
    for (int i = 1, l = 0, r = 0;i <= m;i++) {
        if (i <= r) p[i] = min(z[i - l + 1], r - i + 1);
        while (1 + p[i] <= n && i + p[i] <= m && s[1 + p[i]] == t[i + p[i]]) p[i]++;
        if (i + p[i] - 1 > r) l = i, r = i + p[i] - 1;
    }
    return p;
}
```



### 字符串哈希

```c++
//位权法: 倒序计算 如174855可以看作是558471,子串748即(558470-550000)/xp[1]
const int N = 2e5 + 10;
#define ENABLE_DOUBLE_HASH
const int X1 = 131;
const int X2 = 13331;
const int p1 = 1e9 + 7;//const int p1 = rnd(1e8, 1e9);
const int p2 = 1e9 + 9;//const int p2 = rnd(1e9 + 1, 2e9);
ull xp1[N], xp2[N], xp[N];
void init() {
    xp1[0] = xp2[0] = xp[0] = 1;
    for (int i = 1;i < N;i++) {
        xp1[i] = xp1[i - 1] * X1 % p1;
        xp2[i] = xp2[i - 1] * X2 % p2;
        xp[i] = xp[i - 1] * X1;
    }
}
struct String {
    string s;
    int size, subsize;
    bool sorted;
    vector<ull> h, hl;
    String(string S) :s(S) {
        sorted = subsize = 0;
        size = (int)s.size() - 1;
        h.resize(size + 2);hl.resize(size + 2);
        ull res1 = 0, res2 = 0;
        h[size + 1] = 0;
        for (int j = size;j >= 1;j--) {
#ifdef ENABLE_DOUBLE_HASH
            res1 = (res1 * X1 + s[j]) % p1;
            res2 = (res2 * X2 + s[j]) % p2;
            h[j] = (res1 << 32) | res2;
#else
            res1 = res1 * X1 + s[j];
            h[j] = res1;
#endif
        }
    }
    String(string S, bool t) :s(S) {//这里的t没有用,只是为了和上面区分.构造字符串的反向哈希值
        sorted = subsize = 0;
        size = (int)s.size() - 1;
        h.resize(size + 2);hl.resize(size + 2);
        ull res1 = 0, res2 = 0;
        h[size + 1] = 0;
        for (int j = size;j >= 1;j--) {
#ifdef ENABLE_DOUBLE_HASH
            res1 = (res1 * X1 + s[size - j + 1]) % p1;
            res2 = (res2 * X2 + s[size - j + 1]) % p2;
            h[j] = (res1 << 32) | res2;
#else
            res1 = res1 * X1 + s[size - j + 1];
            h[j] = res1;
#endif
        }
    }
    //获取子串哈希，左闭右闭区间. right-left+1>=0
    ull subs(int left, int right)const {
        int len = right - left + 1;
#ifdef ENABLE_DOUBLE_HASH
        unsigned int mask32 = ~(0u);//111..111
        ull left1 = h[left] >> 32, right1 = h[right + 1] >> 32;
        ull left2 = h[left] & mask32, right2 = h[right + 1] & mask32;
        return(((left1 - right1 * xp1[len] % p1 + p1) % p1) << 32) | (((left2 - right2 * xp2[len] % p2 + p2) % p2));
#else
        return h[left] - h[right + 1] * xp[len];
#endif
    }
    void get_all_subs(int sublen) {
        subsize = size - sublen + 1;
        for (int i = 1;i <= subsize;i++) {
            hl[i] = subs(i, i + sublen - 1);
        }
        sorted = 0;
    }
    void sort_subs() {
        if (!sorted) sort(hl.begin() + 1, hl.begin() + subsize + 1);
        sorted = 1;
    }
    bool match(ull key)const {
        if (!sorted)assert(0);
        if (!subsize)return 0;
        return binary_search(hl.begin() + 1, hl.begin() + subsize + 1, key);
    }
};
//a[ai...]与b[bi...]的最长公共前缀
int LCP(const String& a, const String& b, int ai, int bi) {
    int l = 0, r = min(a.size - ai + 1, b.size - bi + 1);int ans = 0;
    while (l <= r) {
        int mid = l + r >> 1;
        if (a.subs(ai, ai + mid - 1) == b.subs(bi, bi + mid - 1)) {
            l = mid + 1;
            ans = mid;
        }
        else {
            r = mid - 1;
        }
    }
    return ans;
}
//检查S的所有长度len的子串是否都在T中出现
int check(const String& S, String& T, int len) {
    if (T.size < len) return 0;
    T.get_all_subs(len);T.sort_subs();
    for (int i = 1;i + len - 1 <= S.size;i++) {
        if (!T.match(S.subs(i, i + len - 1))) return 0;
    }
    return 1;
}
ull add(const String& a, int l1, int r1, const String& b, int l2, int r2) {
    ull suba = a.subs(l1, r1), subb = b.subs(l2, r2);
#ifdef ENABLE_DOUBLE_HASH
    unsigned int mask32 = ~(0u);//111..111
    ull a1 = suba >> 32, b1 = subb >> 32;
    ull a2 = suba & mask32, b2 = subb & mask32;
    return(((a1 + b1 * xp1[r1 - l1 + 1] % p1 + p1) % p1) << 32) | (((a2 + b2 * xp2[r1 - l1 + 1] % p2 + p2) % p2));
#else 
    return suba + subb * xp[r1 - l1 + 1];
#endif
}

//别忘 s=' '+s;

```

### 封装双哈希

```cpp
struct pxm_hash{
    int n;
    string str;
    vector<int> p, h;
    vector<int> p1, h1; 
    int Mod = 1e9 + 7;
    int P = 13331;
    int P1 = 100153;
    int Mod1 = 998244353;
    pxm_hash(int _n, string _str) : n(_n), str(' ' + _str){ init();}
    pxm_hash(string _str) : n(_str.size()), str(' ' + _str) {init();}
    pxm_hash(){}
    void set_mod(int _Mod){this->Mod = _Mod;}
    void set_P(int _P){this->P = _P;}
    void init(){
        p.resize(n + 1);
        h.resize(n + 1);
        p[0] = 1;
        for(int i = 1; i <= n; i ++)
        {
            h[i] = (h[i - 1] * P % Mod + str[i] - 'a' + 1) % Mod;
            p[i] = p[i - 1] * P % Mod;
        }
    }
    void init_double(){
        p1.resize(n + 1);
        h1.resize(n + 1);
        p1[0] = 1;
        for(int i = 1; i <= n; i ++)
        {
            h1[i] = (h1[i - 1] * P1 % Mod1 + str[i] - 'a' + 1) % Mod1;
            p1[i] = p1[i - 1] * P1 % Mod1;
        }
    }
    int get_hash_single(int l, int r){
        return (h[r] - h[l - 1] * p[r - l + 1] % Mod + Mod) % Mod;
    }
    pair<int, int> get_hash_double(int l, int r){
        return {(h[r] - h[l - 1] * p[r - l + 1] % Mod + Mod) % Mod, (h1[r] - h1[l - 1] * p1[r - l + 1] % Mod1 + Mod1) % Mod1};
    }
};
```



### 线段树维护字符串哈希

```c++
const int N = 2e5 + 10;
const int X1 = 131;
const int X2 = 13331;
const int p1 = 1e9 + 7;//const int p1 = rnd(1e8, 1e9);
const int p2 = 1e9 + 9;//const int p2 = rnd(1e9 + 1, 2e9);
ull xp1[N], xp2[N];
ull pre_xp1[N], pre_xp2[N];
void init() {
    xp1[0] = xp2[0] = 1;
    pre_xp1[0] = pre_xp2[0] = 1;
    for (int i = 1;i < N;i++) {
        xp1[i] = (xp1[i - 1] * X1) % p1;
        pre_xp1[i] = (pre_xp1[i - 1] + xp1[i]) % p1;
        xp2[i] = (xp2[i - 1] * X2) % p2;
        pre_xp2[i] = (pre_xp2[i - 1] + xp2[i]) % p2;
    }
}
//封装·线段树
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Tag {
        char ch;
    };
    struct Info {
        int sz;
        int l, r;
        ull res1 = 0, res2 = 0;//正字符串的第一模数和第二模数
        ull res3 = 0, res4 = 0;//反字符串的第一模数和第二模数
    };
    struct node {
        Info info;
        Tag tag;
    };
    Info friend operator +(const Info& l, const Info& r) {
        int mid = l.l + r.r >> 1;
        return { l.sz + r.sz,l.l,r.r,
        (l.res1 + r.res1 * xp1[l.sz]) % p1,
        (l.res2 + r.res2 * xp2[l.sz]) % p2,
        (r.res3 + l.res3 * xp1[r.sz]) % p1,
        (r.res4 + l.res4 * xp2[r.sz]) % p2
        };
    }
    Info friend operator +(const Info& info, const Tag& tag) {
        return { info.sz,info.l,info.r,
            (tag.ch * pre_xp1[info.sz - 1] % p1),
            (tag.ch * pre_xp2[info.sz - 1] % p2),
            (tag.ch * pre_xp1[info.sz - 1] % p1),
            (tag.ch * pre_xp2[info.sz - 1] % p2)
        };
    }
    Tag friend operator+(const Tag& tag1, const Tag& tag2) {
        return { tag2.ch };
    }
    int n;
    vector<node> tr;
    Segtree(const string& s, int n) :n(n) {
        tr.resize(n << 2);
        build(s, 1, 1, n);
    }
    void build(const string& s, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        tr[x].info.l = l;
        tr[x].info.r = r;
        if (l == r) {
            tr[x].info = { 1,l,r,1ull * s[l],1ull * s[l],1ull * s[l],1ull * s[l] };
            tr[x].tag = { 0 };
            return;
    }
        else {
            int mid = l + r >> 1;
            build(s, ls, l, mid);
            build(s, rs, mid + 1, r);
            pushup(x);
        }
}
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
    void settag(int x, Tag tag) {
        tr[x].info = tr[x].info + tag;
        tr[x].tag = tr[x].tag + tag;
    }
    void pushdown(int x) {//下传标记
        if (tr[x].tag.ch) {
            settag(ls, tr[x].tag);
            settag(rs, tr[x].tag);
            tr[x].tag.ch = 0;
        }
    }
    //l,r代表操作区间
    void update(int x, int l, int r, int ql, int qr, Tag tag) {
        if (l == ql && r == qr) {
            settag(x, tag);
            return;
        }
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) update(ls, l, mid, ql, qr, tag);
        else if (mid < ql) update(rs, mid + 1, r, ql, qr, tag);
        else {
            update(ls, l, mid, ql, mid, tag);
            update(rs, mid + 1, r, mid + 1, qr, tag);
        }
        pushup(x);
    }

    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }

};
```

### 带修字符串哈希

```c++
struct Tree_hash{
    const int k1 = 137, k2 = 173;
    #define ls l, mid, rt << 1
    #define rs mid + 1, r, rt << 1 | 1
    string q;int Pow1[MAXN], Pow2[MAXN],n;
    struct node
    {
        int l, r, len;
        int Key1, Key2;
      
    } tree[MAXN << 2];
    void init(string &s){//s的下标从0开始
        q ='0'+ s;
        n = s.size();
        Pow1[0] = 1;
        Pow2[0] = 1;
        for (int i = 1; i <= n; i++)
        {
            Pow1[i] = Pow1[i - 1] * k1;
            Pow2[i] = Pow2[i - 1] * k2;
        }
        build(1, n, 1);
    }
    void push_up(int L, int R, int rt)
    {
        tree[rt].Key1 = tree[rt << 1].Key1 + tree[rt << 1 | 1].Key1 * Pow1[L];
        tree[rt].Key2 = tree[rt << 1].Key2 + tree[rt << 1 | 1].Key2 * Pow2[L];
    }
    void build(int l, int r, int rt)
    {
        tree[rt].l = l;
        tree[rt].r = r;
        tree[rt].len = r - l + 1;
        tree[rt].Key1 = 0;
        tree[rt].Key2= 0;
        if (l == r)
        {
            tree[rt].Key1 = q[l];
            tree[rt].Key2 = q[l];
            return;
        }
        int mid = (l + r) >> 1;
        build(ls);
        build(rs);
        push_up(mid - l + 1, r - mid, rt);
    }

    void update(int a, char b, int l, int r, int rt) // update(pos, char, 1, n, 1);
    {
        if (l == r)
        {
            tree[rt].Key1 = b;
            tree[rt].Key2 = b;
            return;
        }
        int mid = (l + r) >> 1;
        if (mid >= a)
            update(a, b, ls);
        if (mid < a)
            update(a, b, rs);
        push_up(mid - l + 1, r - mid, rt);
    }
    node query(int x, int y, int l, int r, int rt) //query(ql, qr, 1, n, 1);
    {
        if (l >= x && r <= y)
            return tree[rt];
        node ans, ans1, ans2;
        int T = 0;
        int mid = (l + r) >> 1;
        if (mid >= x)
            ans1 = query(x, y, ls), T += 1;
        if (mid < y)
            ans2 = query(x, y, rs), T += 2;
        if (T == 1)
            ans = ans1;
        else if (T == 2)
            ans = ans2;
        else
        {
            ans.Key1 = ans1.Key1 + ans2.Key1 * Pow1[mid - max(x, l) + 1];
            ans.Key2 = ans1.Key2 + ans2.Key2 * Pow2[mid - max(x, l) + 1];

        }
        return ans;
    }
    pair<int,int>get(int l,int r){//返回l到r之间的双哈希值
        node pl = query(l, r, 1, n, 1);
        return {pl.Key1, pl.Key2};
    }
};
```



### AC自动机

```c++
const int N = 2e5 + 10;
struct ACAM {
    //basic
    static int nxt[N][26];//Trie图边
    inline static int fail[N], root, tot;//fail边(字典串中与当前串的后缀匹配的最长的前缀),根,结点总数
    //extend
    inline static int len[N];//前缀串的长度
    inline static int cnt[N];//数量
    inline static int uid[N];//id
    inline static int deg[N];//度
    static vector<int> adj[N];//邻接表
    map<int, int> mp;int qid;//插入串的编号


    ACAM() { clear(); }
    void clear() {
        memset(nxt[0], 0, sizeof nxt[0]);
        for (int i = 0;i <= tot;i++) adj[i].clear();
        qid = root = tot = fail[0] = len[0] = 0;
        mp.clear();
    }
    int newnode() {
        tot++;
        memset(nxt[tot], 0, sizeof nxt[tot]);
        fail[tot] = len[tot] = cnt[tot] = 0;
        return tot;
    }
    void insert(const string& s) {
        int now = root;
        for (int i = 1;i < s.size();i++) {
            int id = s[i] - 'a';
            if (!nxt[now][id]) nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now] + 1;
            now = nxt[now][id];
        }
        //extend
        cnt[now] += 1;
        uid[now] = ++qid;
        mp[qid] = now;
    }

    void build() {
        //bfs求fail.跳fail链的过程被压缩
        queue<int> q;
        for (int i = 0; i < 26; i++) if (nxt[0][i]) q.push(nxt[0][i]);
        while (q.size()) {
            int u = q.front();q.pop();
            for (int i = 0; i < 26; i++) {
                int v = nxt[u][i];
                if (v) {
                    fail[v] = nxt[fail[u]][i], q.push(v);
                    adj[v].push_back(fail[v]), deg[fail[v]] += 1;//建边拓扑排序
                    //st[v] |= st[fail[v]];//fail链上信息继承,表示该串所有后缀的信息
                }
                else nxt[u][i] = nxt[fail[u]][i];
            }
        }
    }

    void query(string s) {
        int now = root;
        int res = 0;
        for (int i = 1;i < s.size();i++) {
            int id = s[i] - 'a';
            now = nxt[now][id];
            //do something
        }
    }
    void run() {
        //拓扑排序更新答案
        queue<int> q;
        for (int i = 1;i <= tot;i++) {
            if (deg[i] == 0) q.push(i);
        }
        while (q.size()) {
            auto u = q.front();q.pop();
            for (auto v : adj[u]) {
                //do something
                if (--deg[v] == 0) q.push(v);
            }
        }
    }
};
int ACAM::nxt[N][26];
vector<int> ACAM::adj[N];

//s=' '+s; build()

```



### KMP

```c++
struct KMP {
    string s;
    int len;
    vector<int> nxt;//S[1,i]的非平凡最大Border
    KMP(string s) :s(s) {
        len = s.size() - 1;
        nxt.resize(len + 1);
        init(s);
    }
    void init(string s) {
        for (int i = 2;i <= len;i++) {
            nxt[i] = nxt[i - 1];
            while (nxt[i] && s[i] != s[nxt[i] + 1]) nxt[i] = nxt[nxt[i]];
            nxt[i] += (s[i] == s[nxt[i] + 1]);
        }
    }
    //s在ss中出现的位置
    vector<int> match(string ss, int ONLY_ONE = 0) {
        int len_ss = ss.size() - 1;
        vector<int> res;
        for (int i = 1, j = 1;i <= len_ss;) {
            while (j != 1 && ss[i] != s[j]) j = nxt[j - 1] + 1;
            if (ss[i] == s[j]) i++, j++;
            else i++;
            if (j == len + 1) {
                res.push_back(i - j + 1);
                if (ONLY_ONE) return res;
                j = nxt[len] + 1;
            }
        }
        return res;
    }
    //周期
    vector<int> period() {
        vector<int> res;
        int now = len;
        while (now) {
            now = nxt[now];
            res.push_back(len - now);
        }
        return res;
    }
    //循环节
    vector<int> loop() {
        vector<int> res;
        int now = len;
        for (auto i : period()) {
            if (len % i == 0) res.push_back(i);
        }
        return res;
    }

};
//s=' '+s ,ss=' '+ss  !!!
```



### manacher

```c++
struct Manacher {
    int n;//扩展后的长度
    string s;//扩展后的回文串
    vector<int> rad;//i为中心的最大回文半径
    
    char operator[](int k)const { return s[k]; }
    char& operator[](int k) { return s[k]; }
    
    vector<int> rad;//i为中心的最大回文半径
    Manacher(string S) {
        int len = S.size() - 1;
        n = 2 * len + 1;
        s.resize(n + 1);
        rad.resize(n + 1);
        init(S);
        manacher();
    }
    void init(string S) {
        int len = S.size() - 1;
        s[0] = ' ';
        s[n] = '#';
        for (int i = len;i >= 1;i--) {
            s[i * 2] = S[i];s[i * 2 - 1] = '#';
        }
    }
    void manacher() {
        rad[1] = 1;int k = 1;
        for (int i = 2;i <= n;i++) {
            int p = k + rad[k] - 1;
            if (i <= p)  rad[i] = min(rad[2 * k - i], p - i + 1);
            else rad[i] = 1;
            while (i + rad[i] <= n && s[i + rad[i]] == s[i - rad[i]]) rad[i]++;
            if (i + rad[i] > k + rad[k]) k = i;
        }
    }
};
//注意添加 s=' '+s;

```



### manacher2

```c++
void manacher(string &s, vector<int> &dp)
{ // manacher求回文半径,dp[i]-1即为当前位置回文串长度
    string ss = "&";
    for (auto &i : s)
    {
        ss += '#';
        ss += i;
    }
    ss += "#@";
    int right = 0, pos = 0;
    dp.resize(ss.size());
    for (int i = 1; i < ss.size(); i++)
    {
        if (i < right)
            dp[i] = min(dp[2 * pos - i], right - i);
        else
            dp[right = i] = 1;
        while (ss[i - dp[i]] == ss[i + dp[i]])
            dp[i]++;
        if (i + dp[i] > right)
            right = i + dp[i], pos = i;
    }
}
```



### 回文自动机

```c++
const int N = 2e5 + 10;
struct PAM {
    inline static int s[N], now;//字符串,当前位置
    static int nxt[N][26]; inline static int fail[N], faildep[N], len[N], last, tot, sum;
    //边,fail边,该节点代表的回文串长度,上一个字母所在节点,结点总数,回文串数
    inline static int num[N];//所有回文串出现次数
    void clear() {
        //奇数长度的root为节点1,偶数长度的root为节点0
        s[0] = -1;
        len[0] = 0, len[1] = -1;
        fail[0] = 1, fail[1] = 0;
        faildep[0] = faildep[1] = 0;
        tot = 1;now = last = 0;
        sum = 0;
        memset(nxt[0], 0, sizeof nxt[0]);
        memset(nxt[1], 0, sizeof nxt[1]);
    }
    PAM() { clear(); }
    PAM(string S) { clear();init(S);build(); }
    int newnode(int Len) {
        tot++;
        memset(nxt[tot], 0, sizeof nxt[tot]);
        fail[tot] = num[tot] = 0;
        len[tot] = Len;
        return tot;
    }
    int jump_fail(int x) {
        //跳fail链(Border链,回文后缀链)直到满足添加当前字符仍为回文.
        while (s[now - len[x] - 1] != s[now]) x = fail[x];
        return x;
    }
    void add(int ch) {
        s[++now] = ch;
        int cur = jump_fail(last);
        if (!nxt[cur][ch]) {
            int t = newnode(len[cur] + 2);
            int from = nxt[jump_fail(fail[cur])][ch];
            fail[t] = from;
            faildep[t] = faildep[from] + 1;
            nxt[cur][ch] = t;
        }
        last = nxt[cur][ch];num[last]++;
        sum += faildep[last];
    }
    void build() {
        //fail[i]<i,可以直接大到小扫描
        for (int i = tot;i >= 2;i--) {
            num[fail[i]] += num[i];
        }
        num[0] = num[1] = 0;
    }
    void init(string s) {
        for (int i = 1;i < (int)s.size();i++) {
            add(s[i] - 'a');
        }
    }
};
int PAM::nxt[N][26];
//记得s=' '+s;   !!!
```

### 后缀数组

```c++
const int N = 2e5 + 10;
struct SA {
    int n, m;//字符串长度,字符集大小
    inline static int s[N];//字符串
    inline static int cnt[N], sa_second[N], oldrank[N];//计数,第二关键字排名为i的后缀,旧排名
    vector<int> sa, rank, height;//第i名的后缀,后缀i的排名,排名为i与排名为i-1的LCP
    ST table;

    bool check(int x, int y, int w) {
        return oldrank[x] == oldrank[y] && x <= n && y <= n && oldrank[x + w] == oldrank[y + w];
    }
    SA(string S) {
        n = S.size() - 1;
        sa.resize(n + 1), rank.resize(n + 1), height.resize(n + 1);
        m = 128;
        for (int i = 1;i <= n;i++) s[i] = S[i];
        getSA();
        getHeight();
    }
    void getHeight() {
        int k = 0;
        for (int i = 1;i <= n;i++) {
            if (k) k--;
            int j = sa[rank[i] - 1];
            while (i + k <= n && j + k <= n && s[i + k] == s[j + k]) k++;
            height[rank[i]] = k;
        }
        table.init(height);
    }
    void getSA() {
        for (int i = 0;i <= m;i++) cnt[i] = 0;
        for (int i = 1;i <= n;i++) cnt[rank[i] = s[i]]++;
        for (int i = 1;i <= m;i++) cnt[i] += cnt[i - 1];
        for (int i = n;i >= 1;i--) sa[cnt[rank[i]]--] = i;

        //双关键字基数排序,每次倍增添加关键字
        for (int w = 1, p;w <= n;w <<= 1, m = p) {
            p = 0;
            //尾部第二关键字为空的元素,排名设为最小
            for (int i = n;i > n - w;i--) sa_second[++p] = i;

            for (int i = 1;i <= n;i++) {
                if (sa[i] > w) sa_second[++p] = sa[i] - w;
            }
            for (int i = 0;i <= m;i++) cnt[i] = 0;
            for (int i = 1;i <= n;i++) cnt[rank[sa_second[i]]]++;
            for (int i = 1;i <= m;i++) cnt[i] += cnt[i - 1];
            for (int i = n;i >= 1;i--) sa[cnt[rank[sa_second[i]]]--] = sa_second[i];

            for (int i = 0;i <= n;i++) oldrank[i] = rank[i];
            rank[sa[1]] = p = 1;
            for (int i = 2;i <= n;i++) {
                rank[sa[i]] = check(sa[i], sa[i - 1], w) ? p : ++p;
            }
            if (p == n) {
                for (int i = 1; i <= n; i++) sa[rank[i]] = i;
                break;
            }
        }
    }
    int lcp(int x, int y) {
        int rkx = rank[x], rky = rank[y];
        if (rkx > rky) swap(rkx, rky);
        rkx++;
        return table.query(rkx, rky);
    }
};

//记得s=' '+s !!!
```





## 数学

### 类欧几里得

``` cpp
using ull = unsigned long long;
ull floor_sum(ull n, ull m, ull a, ull b) {
    ull ans = 0;
    for (;;) {
        if (a >= m) ans += n * (n - 1) / 2 * (a / m), a %= m;
        if (b >= m) ans += n * (b / m), b %= m;
        ull ymax = a * n + b;//use i128 if it's big
        if (ymax < m) break;
        n = ymax / m;
        b = ymax % m;
        swap(m, a);
    }
    return ans;
}
```



### 多项式

```c++
namespace Cipolla {
    int mul(int x, int y) { return 1ll * x * y % MOD; }
    uint qp(uint a, int b) { uint res = 1; for (; b; b >>= 1, a = mul(a, a))  if (b & 1)  res = mul(res, a); return res; }
    int sqr_i;
    struct spc_Cp {
        int x, y;
        spc_Cp() { ; }
        spc_Cp(int x, int y) : x(x), y(y) {}
        inline spc_Cp operator * (const spc_Cp& t) const { return (spc_Cp) { (mul(x, t.x) + mul(mul(y, t.y), sqr_i)) % MOD, (mul(x, t.y) + mul(y, t.x)) % MOD }; }
    };
    spc_Cp qp(spc_Cp a, int b) {
        spc_Cp res = spc_Cp(1, 0);
        while (b) {
            if (b & 1) res = res * a;
            b >>= 1, a = a * a;
        }
        return res;
    }
    //解是res和MOD-res
    int Cipolla(int n) {
        srand(time(NULL));
        if (qp(n, MOD >> 1) == MOD - 1) return -1;
        ll t = mul(rand(), rand());
        while (qp((mul(t, t) - n) % MOD + MOD, MOD >> 1) == 1) t = 1ll * rand() * rand() % MOD;//找到非二次剩余的数,期望循环次数为2
        sqr_i = ((mul(t, t) - n) % MOD + MOD) % MOD;
        int res = qp(spc_Cp(t, 1), MOD + 1 >> 1).x;
        //return res;//返回任何一个解
        return min(res, MOD - res);//返回较小解
    }
}


int rev[N];//NTT/FFT反转二进制
namespace MTT {//任意模数多项式乘法
    const double PI = acos((double)-1);
    struct Cp {
        double x, y;
        Cp() { ; }
        Cp(double _x, double _y) : x(_x), y(_y) { }
        inline Cp operator + (const Cp& t) const { return (Cp) { x + t.x, y + t.y }; }
        inline Cp operator - (const Cp& t) const { return (Cp) { x - t.x, y - t.y }; }
        inline Cp operator * (const Cp& t) const { return (Cp) { x* t.x - y * t.y, x* t.y + y * t.x }; }
    }A[N], B[N], C[N], w[N / 2];

#define E(x) ll(x+0.5)%P

    void FFT(int n, Cp* a, int f) {
        for (int i = 0;i <= n - 1;i++) if (rev[i] < i) swap(a[i], a[rev[i]]);
        w[0] = Cp(1, 0);
        for (int i = 1;i < n;i <<= 1) {
            Cp t = Cp(cos(PI / i), f * sin(PI / i));
            for (int j = i - 2;j >= 0;j -= 2) w[j + 1] = t * (w[j] = w[j >> 1]);
            for (int l = 0;l < n;l += 2 * i) {
                for (int j = l;j < l + i;j++) {
                    Cp t = a[j + i] * w[j - l];
                    a[j + i] = a[j] - t;
                    a[j] = a[j] + t;
                }
            }
        }
        if (f == -1) for (int i = 0;i <= n - 1;i++) a[i].x /= n, a[i].y /= n;
    }

    void Multiply(vector<uint> a, vector<uint> b, vector<uint>& res, int P) {
        // [0,n-1]*[0,m-1]->[0,n+m-2]
        int n = a.size(), m = b.size();res.resize(n + m - 1);
        int S = (1 << 15) - 1;

        int R = 1, cc = -1;
        while (R <= n + m - 1) R <<= 1, cc++;
        for (int i = 1;i <= R;i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << cc);
        for (int i = 0;i <= n - 1;i++) A[i] = Cp((a[i] & S), (a[i] >> 15));
        for (int i = 0;i <= m - 1;i++) B[i] = Cp((b[i] & S), (b[i] >> 15));
        for (int i = n;i <= R - 1;i++) A[i] = Cp(0, 0);
        for (int i = m;i <= R - 1;i++) B[i] = Cp(0, 0);

        FFT(R, A, 1), FFT(R, B, 1);
        for (int i = 0;i <= R - 1;i++) {
            int j = (R - i) % R;
            C[i] = Cp((A[i].x + A[j].x) / 2, (A[i].y - A[j].y) / 2) * B[i];
            B[i] = Cp((A[i].y + A[j].y) / 2, (A[j].x - A[i].x) / 2) * B[i];
        }
        FFT(R, C, -1), FFT(R, B, -1);
        for (int i = 0;i <= n + m - 2;i++) {
            ll a = E(C[i].x), b = E(C[i].y), c = E(B[i].x), d = E(B[i].y);
            res[i] = (a + ((b + c) << 15) + (d << 30)) % P;
        }
    }
    // void Multiply_db(vector<double> a, vector<double> b, vector<double>& res) {//答案double类型
    //     // [0,n-1]*[0,m-1]->[0,n+m-2]
    //     int n = a.size(), m = b.size();res.resize(n + m - 1);

    //     int R = 1, cc = -1;
    //     while (R <= n + m - 1) R <<= 1, cc++;
    //     for (int i = 1;i <= R;i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << cc);
    //     for (int i = 0;i <= n - 1;i++) A[i] = Cp(a[i], 0);
    //     for (int i = 0;i <= m - 1;i++) B[i] = Cp(b[i], 0);
    //     for (int i = n;i <= R - 1;i++) A[i] = Cp(0, 0);
    //     for (int i = m;i <= R - 1;i++) B[i] = Cp(0, 0);

    //     FFT(R, A, 1), FFT(R, B, 1);
    //     for (int i = 0;i < R;i++) C[i] = A[i] * B[i];
    //     FFT(R, C, -1);
    //     for (int i = 0;i <= n + m - 2;i++) res[i] = C[i].x;
    // }

#undef E
}

int Add(int x, int y) { return (x + y >= MOD) ? x + y - MOD : x + y; }
int Dec(int x, int y) { return (x - y < 0) ? x - y + MOD : x - y; }
int mul(int x, int y) { return 1ll * x * y % MOD; }
uint qp(uint a, int b) { uint res = 1; for (; b; b >>= 1, a = mul(a, a))  if (b & 1)  res = mul(res, a); return res; }


namespace NTT {
    int sz;
    uint w[2500005], w_mf[2500005];
    int mf(int x) { return (1ll * x << 32) / MOD; }
    void init(int n) {
        for (sz = 2; sz < n; sz <<= 1);
        uint pr = qp(3, (MOD - 1) / sz);
        w[sz / 2] = 1; w_mf[sz / 2] = mf(1);
        for (int i = 1; i < sz / 2; i++)  w[sz / 2 + i] = mul(w[sz / 2 + i - 1], pr), w_mf[sz / 2 + i] = mf(w[sz / 2 + i]);
        for (int i = sz / 2 - 1; i; i--)  w[i] = w[i << 1], w_mf[i] = w_mf[i << 1];
    }
    void ntt(vector<uint>& A, int L) {
        for (int d = L >> 1; d; d >>= 1)
            for (int i = 0; i < L; i += (d << 1))
                for (int j = 0; j < d; j++) {
                    uint x = A[i + j] + A[i + d + j];
                    if (x >= 2 * MOD)  x -= 2 * MOD;
                    ll t = A[i + j] + 2 * MOD - A[i + d + j], q = t * w_mf[d + j] >> 32; int y = t * w[d + j] - q * MOD;
                    A[i + j] = x; A[i + d + j] = y;
                }
        for (int i = 0; i < L; i++)  if (A[i] >= MOD)  A[i] -= MOD;
    }
    void intt(vector<uint>& A, int L) {
        for (int d = 1; d < L; d <<= 1)
            for (int i = 0; i < L; i += (d << 1))
                for (int j = 0; j < d; j++) {
                    uint x = A[i + j]; if (x >= 2 * MOD)  x -= 2 * MOD;
                    ll t = A[i + d + j], q = t * w_mf[d + j] >> 32, y = t * w[d + j] - q * MOD;
                    A[i + j] = x + y; A[i + d + j] = x + 2 * MOD - y;
                }
        int k = (L & (-L));
        reverse(A.begin() + 1, A.end());
        for (int i = 0; i < L; i++) {
            ll m = -A[i] & (L - 1);
            A[i] = (A[i] + m * MOD) / k;
            if (A[i] >= MOD)  A[i] -= MOD;
        }
    }
}
int _inv[N];
void Poly_init(int mod = MOD) {
    _inv[1] = 1;
    for (int i = 2;i < N;i++) _inv[i] = 1llu * _inv[MOD % i] * (MOD - MOD / i) % MOD;
}
struct Poly {
    vector<uint> p;
    Poly(int n) { p.resize(n); }
    Poly(int n, int k) { p.resize(n);for (int i = 0;i < n;i++) p[i] = k; }
    Poly() {}
    uint operator[](const int& k)const { return p[k]; }
    uint& operator[](const int& k) { return p[k]; }
    Poly extend(int x) { Poly c = *this;c.p.resize(x);return c; }
    int deg() { return (int)p.size() - 1; }
    void resize(int n) { p.resize(n); }
    int size() { return p.size(); }
    void rev() { reverse(p.begin(), p.end()); }
};
Poly operator+ (Poly A, Poly B) {
    int n = A.size(), m = B.size();
    Poly c(max(n, m));
    for (int i = 0; i < n; i++)  c[i] = A[i];
    for (int i = 0; i < m; i++)  c[i] = Add(c[i], B[i]);
    return c;
}
Poly operator- (Poly A, Poly B) {
    int n = A.size(), m = B.size();
    Poly c(max(n, m));
    for (int i = 0; i < n; i++)  c[i] = A[i];
    for (int i = 0; i < m; i++)  c[i] = Dec(c[i], B[i]);
    return c;
}
Poly operator*(Poly A, Poly B) {//MOD=998244353,..a*2^k+1
    int n = A.deg() + B.deg() + 1;
    int lim;for (lim = 1; lim < n; lim <<= 1); NTT::init(lim);
    A.resize(lim); B.resize(lim);
    NTT::ntt(A.p, lim); NTT::ntt(B.p, lim);
    for (int i = 0; i < lim; i++)  A[i] = mul(A[i], B[i]);
    NTT::intt(A.p, lim); return A.extend(n);
}

// Poly operator*(Poly A, Poly B) {//任意模数多项式乘法
//     int n = A.deg() + B.deg() + 1;
//     Poly res(n);
//     MTT::Multiply(A.p, B.p, res.p, MOD);
//     return res.extend(n);
// }

// Poly operator*(Poly A, Poly B) {//答案double类型||如果答案不取模,改成longdouble即可,注意最后的结果需要ll(res[i]+0.5)取整
//     int n = A.deg() + B.deg() + 1;
//     Poly res(n);
//     MTT::Multiply_db(A.p, B.p, res.p);
//     return res.extend(n);
// }

Poly Dev(Poly A) {//多项式求导
    int n = A.size();
    for (int i = 1;i < n;i++) A[i - 1] = mul(A[i], i);
    return A[n - 1] = 0, A;
}
Poly Int(Poly A) {//多项式求积分
    int n = A.size();
    for (int i = n - 1;i >= 0;i--) A[i] = mul(A[i - 1], _inv[i]);//预处理逆元降低复杂度
    //for (int i = n - 1;i >= 0;i--) A[i] = mul(A[i - 1], qp(i, MOD - 2));//直接求逆元
    return A[0] = 0, A;
}
Poly Inv(Poly A) {//多项式乘法逆元
    int n = A.size();
    if (n == 1)  return A[0] = qp(A[0], MOD - 2), A;
    Poly B = A; B.resize((n + 1) >> 1); B = Inv(B);
    int lim; for (lim = 1; lim < (n << 1); lim <<= 1); NTT::init(lim);
    A.resize(lim); B.resize(lim);
    NTT::ntt(A.p, lim); NTT::ntt(B.p, lim);
    for (int i = 0; i < lim; i++)  A[i] = mul(Dec(2, mul(A[i], B[i])), B[i]);
    NTT::intt(A.p, lim); return A.extend(n);
}
Poly operator/(Poly A, Poly B) {
    A.rev(), B.rev();
    int n = A.size(), m = B.size();
    A.resize(n - m + 1), B.resize(n - m + 1);
    B = Inv(B);
    Poly C = A * B;C.resize(n - m + 1);C.rev();
    return C;
}
Poly operator%(Poly A, Poly B) {
    Poly C = A / B;
    return (A - (B * C).extend(A.size())).extend((int)B.size() - 1);
}

Poly __Inv(Poly A) {//任意模数多项式乘法逆元
    int n = A.size();
    if (n == 1) return A[0] = qp(A[0], MOD - 2), A;
    Poly B = A;B.resize((n + 1) >> 1); B = __Inv(B).extend(n);
    Poly C(1), D(1);
    MTT::Multiply(A.p, B.p, C.p, MOD);C.resize(n);
    MTT::Multiply(C.p, B.p, D.p, MOD);D.resize(n);
    for (int i = 0;i < n;i++) B[i] = Dec(Add(B[i], B[i]), D[i]);
    return B.extend(n);
}

//保证[x ^ 0]f(x) = 1
Poly Ln(Poly A) {//多项式对数
    Poly B; int n = A.size(); B.resize(n);
    for (int i = 1; i < n; i++)  B[i - 1] = mul(A[i], i); B[n - 1] = 0;
    B = (B * Inv(A)).extend(n);
    B = Int(B);
    return B;
}
//保证[x ^ 0]f(x) = 0
Poly Exp(Poly A) {//多项式指数
    int n = A.size();
    if (n == 1) return A[0] = 1, A;
    Poly B = A; B.resize((n + 1) >> 1); B = Exp(B).extend(n);
    Poly C = Ln(B);
    for (int i = 0; i < n; i++)  C[i] = Dec(A[i], C[i]); C[0] = Add(C[0], 1);
    return (B * C).extend(n);
}
Poly __Exp(Poly A) {//任意模数多项式指数
    int n = A.size();
    if (n == 1) return A[0] = 1, A;
    Poly B = A;B.resize((n + 1) >> 1); B = __Exp(B).extend(n);
    Poly C = Ln(B);
    Poly D(1), E(1);
    MTT::Multiply(B.p, C.p, D.p, MOD);D.resize(n);
    MTT::Multiply(B.p, A.p, E.p, MOD);E.resize(n);
    for (int i = 0;i < n;i++) B[i] = Add(Dec(B[i], D[i]), E[i]);
    return B.extend(n);
}

//保证[x ^ 0]f(x) = 1
Poly Sqrt(Poly A) {//多项式开根
    int n = A.size();
    if (n == 1) return A[0] = 1, A;
    Poly B = A;B.resize((n + 1) >> 1); B = Sqrt(B).extend(n);
    Poly C = Inv(B).extend(n);
    int lim; for (lim = 1; lim < (n << 1); lim <<= 1); NTT::init(lim);
    A.resize(lim); B.resize(lim);C.resize(lim);
    NTT::ntt(A.p, lim);NTT::ntt(B.p, lim);NTT::ntt(C.p, lim);
    for (int i = 0;i < lim;i++) B[i] = mul(mul(Add(mul(B[i], B[i]), A[i]), _inv[2]), C[i]);
    NTT::intt(B.p, lim);
    return B.extend(n);
}
Poly Sqrt_pro(Poly A) {//多项式开根
    int n = A.size();
    if (n == 1) return A[0] = Cipolla::Cipolla(A[0]), A;
    Poly B = A;B.resize((n + 1) >> 1); B = Sqrt_pro(B).extend(n);
    Poly C = (B * B).extend(n);
    for (int i = 0;i < n;i++) B[i] = mul(2, B[i]);
    for (int i = 0;i < n;i++) C[i] = Add(C[i], A[i]);
    C = C * Inv(B);
    return C.extend(n);
}

//保证[x ^ 0]f(x) = 1
//k很大时,可以计算前对k模p (注意数论中是费马小定理,模p-1)
Poly Qpow(Poly A, int k) {//多项式快速幂
    int n = A.size();Poly B = Ln(A);
    for (int i = 0;i < n;i++) B[i] = mul(B[i], k);
    return Exp(B);
}
//k很大时,可以计算前对k模p和p-1记录在k1和k2(注意数论中是费马小定理,模p-1)
Poly Qpow_pro(Poly a, int k) {//任意首项多项式快速幂
    int k1 = k % MOD, k2 = k % (MOD - 1);
    int n = a.size();
    int shift = 0;
    for (int i = 0;i < n && a[i] == 0;i++) shift++;
    if (1ll * shift * k1 >= n) {
        for (int i = 0;i < n;i++) a[i] = 0;
        return a;
    }
    int inv_first = qp(a[shift], MOD - 2);int t = qp(a[shift], k2);
    for (int i = 0;i < n;i++) {
        if (i + shift < n) a.p[i] = mul(a[i + shift], inv_first);
        else a[i] = 0;
    }
    a = Ln(a);
    for (int i = 0;i < n;i++) a[i] = mul(a[i], k1);
    a = Exp(a);
    shift *= k1;
    for (int i = n - 1;i >= shift;i--) a[i] = mul(a[i - shift], t);
    for (int i = 0;i < shift;i++) a[i] = 0;
    return a;
}

//i^2=-1(mod p),对-1用二次剩余算出i
//i = 86583718 (mod 998244353)
//保证[x ^ 0]f(x) = 0
const int I = 86583718;
Poly Sin(Poly A) {//多项式sin
    int n = A.size();
    int inv_2i = qp(mul(2, I), MOD - 2);
    for (int i = 0;i < n;i++) A[i] = mul(A[i], I);
    Poly B = Exp(A), C = Inv(B);
    for (int i = 0;i < n;i++) B[i] = mul(Dec(B[i], C[i]), inv_2i);
    return B;
}
//保证[x ^ 0]f(x) = 0
Poly Cos(Poly A) {//多项式cos
    int n = A.size();
    for (int i = 0;i < n;i++) A[i] = mul(A[i], I);
    Poly B = Exp(A), C = Inv(B);
    for (int i = 0;i < n;i++) B[i] = mul(Add(B[i], C[i]), _inv[2]);
    return B;
}
//保证[x ^ 0]f(x) = 0
Poly Arcsin(Poly A) {
    int n = A.size();
    Poly B = Dev(A);
    A = (A * A).extend(n);
    for (int i = 0;i < n;i++) A[i] = Dec(0, A[i]);A[0] = Add(1, A[0]);
    A = Sqrt(A);
    B = (B * Inv(A)).extend(n);
    B = Int(B);
    return B;
}
//保证[x ^ 0]f(x) = 0
Poly Arccos(Poly A) {
    int n = A.size();
    Poly B = Dev(A);for (int i = 0;i < n;i++) B[i] = Dec(0, B[i]);
    A = (A * A).extend(n);
    for (int i = 0;i < n;i++) A[i] = Dec(0, A[i]);A[0] = Add(1, A[0]);
    A = Sqrt(A);
    B = (B * Inv(A)).extend(n);
    B = Int(B);
    return B;
}
//保证[x ^ 0]f(x) = 0
Poly Arctan(Poly A) {
    int n = A.size();
    Poly B = Dev(A);
    A = (A * A).extend(n);
    A[0] = Add(1, A[0]);
    B = (B * Inv(A)).extend(n);
    B = Int(B);
    return B;
}

Poly Stiring_2_row(int n) {//SC(n,i)
    Poly A(n + 1);for (int i = 0, infact_i = 1;i <= n;i++, infact_i = mul(infact_i, _inv[i])) A[i] = mul(((i & 1) ? MOD - 1 : 1), infact_i);
    Poly B(n + 1);for (int i = 0, infact_i = 1;i <= n;i++, infact_i = mul(infact_i, _inv[i])) B[i] = mul(qp(i, n), infact_i);
    A = A * B;
    return A;
}

Poly Stiring_1_col(int n, int m) {//SA(i,m)
    int infact_m = 1;for (int i = 1;i <= m;i++) infact_m = mul(infact_m, _inv[i]);
    Poly A(n + 1);for (int i = 0;i <= n;i++) A[i] = qp(i, MOD - 2);
    A = Qpow_pro(A, m);
    for (int i = 0, fact_i = 1;i <= n;i++, fact_i = mul(fact_i, i)) A[i] = mul(mul(A[i], infact_m), fact_i);
    return A;
}

//记得Poly_init, 如果仅是乘法则不需要
//Poly读入和初始化时,记得取模. f[i] = -1  ==> f[i] = MOD-1 
//MTT的rev开lim大小,为方便一般3~4倍即可

```



### 数论

#### 全家桶

```c++
#define DF_NT//数论
#define DF_Primes//质数
#define DF_binom//组合数及逆元
#define DF_min25//组合数及逆元
//#define DF_C//组合数递推
//#define DF_LP//光速幂
//#define DF_Stirling//斯特林数递推
//#define DF_Barrett//Barrett约减
//#define DF_Cipolla//二次剩余

namespace MATH {

    int Add(int x, int y) { return (x + y >= MOD) ? x + y - MOD : x + y; }
    int Dec(int x, int y) { return (x - y < 0) ? x - y + MOD : x - y; }
    int mul(int x, int y) { return 1ll * x * y % MOD; }
    int norm(int x, int mod = MOD) { return (x % mod + mod) % mod; }

    ll qp(ll a, ll n, int mod = MOD) {
        a %= mod;//n%=(mod-1); if mod is prime
        ll res = 1;
        while (n) {
            if (n & 1) res = res * a % mod;
            a = a * a % mod;n >>= 1;
        }
        return res;
    }

    namespace exgcd {
        int exgcd(int a, int b, int& x, int& y) {
            if (b == 0) {
                x = 1, y = 0;
                return a;
            }
            int x1, y1, d;
            d = exgcd(b, a % b, x1, y1);
            x = y1, y = x1 - a / b * y1;
            return d;
        }

        int inv(int a, int b = MOD) { int x, y;exgcd(a, b, x, y);return norm(x, b); } //exgcd求逆元
    }

#ifdef DF_min25
    namespace min25 {
        const int SIZE = 1e5 + 10;
        const int EXP_SIZE = 2;
        int primes[SIZE], minFac[SIZE], pfx[SIZE][EXP_SIZE], primesPt, lim;
        ll g[SIZE * EXP_SIZE][EXP_SIZE], dsc[SIZE * EXP_SIZE], inv2, inv3;
        array<int, 2> indx[SIZE];  // indx[x]: index of <x, n / x>
        const array<int, 2> csts[EXP_SIZE] = { {-1, 1}, {1, 2} };

        void initPrimes(int siz) {
            inv2 = qp(2, MOD - 2, MOD); inv3 = qp(3, MOD - 2, MOD);
            fill(minFac + 0, minFac + siz + 1, 0); primesPt = 0;
            for (int i = 2; i <= siz; i++) {
                if (minFac[i] == 0) minFac[i] = i, primes[++primesPt] = i;
                for (int j = 1; j <= primesPt && primes[j] <= min(minFac[i], siz / i); j++) minFac[i * primes[j]] = primes[j];
            }

            //f(p)的多项式每一项的前缀和
            for (int i = 1; i <= primesPt; i++) {
                for (int e = 0; e < EXP_SIZE; e++) {//展开写可能变快
                    pfx[i][e] = Add(pfx[i - 1][e], qp(primes[i], csts[e][1], MOD));
                }
            }
        }

        //计算f(p)
        const auto f = [](ll p) {
            p %= MOD;
            return p * (p - 1) % MOD;
            //ll res = 0;
            // for (int e = 0; e < EXP_SIZE; e++) {//展开写会快很多
            //     res = (res + csts[e][0] * qp(p, csts[e][1], MOD)) % MOD;
            // }
            // return res;
            };

        //i^k前缀和,次数高的话需要拉插
        const auto sum = [](ll n, ll exp) {
            n %= MOD;
            if (exp == 0) return n;
            ll res = n * (n + 1) % MOD * inv2 % MOD;
            if (exp == 2) return res * ((n << 1) + 1) % MOD * inv3 % MOD;
            return res;
            };

        ll sieve(ll x, ll pt, ll n) {
            if (x <= 1 || primes[pt] > x) return 0;
            int k = x <= lim ? indx[x][0] : indx[n / x][1];
            ll res = 0;
            // for (int e = 0; e < EXP_SIZE; e++) {//展开写会快很多
            //     res = (res + csts[e][0] * (g[k][e] - pfx[pt - 1][e]) % MOD + MOD) % MOD;
            // }
            res = Add(res, Dec(pfx[pt - 1][0], g[k][0]));
            res = Add(res, Dec(g[k][1], pfx[pt - 1][1]));

            for (int i = pt; i <= primesPt && 1ll * primes[i] * primes[i] <= x; i++) {
                ll pk = primes[i], pk1 = 1ll * primes[i] * primes[i];
                for (int e = 1; pk1 <= x; pk = pk1, pk1 *= primes[i], e++) {
                    res = (res + f(pk) * sieve(x / pk, i + 1, n) % MOD + f(pk1)) % MOD;
                }
            }

            return (res + MOD) % MOD;
        }

        ll run(ll n) {
            lim = sqrt(n); initPrimes(lim); int dscPt = 0;
            for (ll l = 1, r; l <= n; l = r + 1) {
                r = n / (n / l); ll v = n / l; dsc[dscPt] = v;
                for (int e = 0; e < EXP_SIZE; e++)
                    g[dscPt][e] = sum(dsc[dscPt], csts[e][1]) - 1;
                v <= lim ? indx[v][0] = dscPt : indx[n / v][1] = dscPt; dscPt++;
            }

            for (int i = 1; i <= primesPt; i++) {
                for (int j = 0; j < dscPt && 1ll * primes[i] * primes[i] <= dsc[j]; j++) {
                    ll v = dsc[j] / primes[i];
                    int k = v <= lim ? indx[v][0] : indx[n / v][1];

                    // for (int e = 0; e < EXP_SIZE; e++) {//展开写会快很多
                    //     g[j][e] = (g[j][e] - qp(primes[i], csts[e][1], MOD) * (g[k][e] - pfx[i - 1][e] + MOD) % MOD + MOD) % MOD;
                    // }

                    g[j][0] = Dec(g[j][0], mul(primes[i], Dec(g[k][0], pfx[i - 1][0])));
                    g[j][1] = Dec(g[j][1], mul(mul(primes[i], primes[i]), Dec(g[k][1], pfx[i - 1][1])));

                }
            }

            return Add(sieve(n, 1, n), 1);
        }
    }
#endif

#ifdef DF_Barrett
    //Barrett约减
    namespace FastMod {
        struct FastMod {
            ull b, m;
            FastMod(ull b) : b(b), m(ull((__uint128_t(1) << 64) / b)) {}
            ull reduce(ull a) {
                ull q = (ull)((__uint128_t(m) * a) >> 64);
                ull r = a - q * b; // can be proven that 0 <= r < 2*b
                return r >= b ? r - b : r;
            }
        };
        FastMod FAST_MOD(MOD);

        const __int128 ONE = 1;
        int qp(int a, int n, int mod = MOD) {
            a %= mod;
            int res = 1;
            while (n) {
                if (n & 1)  res = ONE * res * a % mod;
                a = ONE * a * a % mod;n >>= 1;
            }
            return res;
        }
        int mul(int x, int y, int mod = MOD) { return FAST_MOD.reduce(x * y); } // Barrett取模
    }
#endif

#ifdef DF_Cipolla
    //二次剩余
    namespace Cipolla {
        int mul(int x, int y) { return 1ll * x * y % MOD; }
        uint qp(uint a, int b) { uint res = 1; for (; b; b >>= 1, a = mul(a, a))  if (b & 1)  res = mul(res, a); return res; }
        int sqr_i;
        struct spc_Cp {
            int x, y;
            spc_Cp() { ; }
            spc_Cp(int x, int y) : x(x), y(y) {}
            inline spc_Cp operator * (const spc_Cp& t) const { return (spc_Cp) { (mul(x, t.x) + mul(mul(y, t.y), sqr_i)) % MOD, (mul(x, t.y) + mul(y, t.x)) % MOD }; }
        };
        spc_Cp qp(spc_Cp a, int b) {
            spc_Cp res = spc_Cp(1, 0);
            while (b) {
                if (b & 1) res = res * a;
                b >>= 1, a = a * a;
            }
            return res;
        }
        //解是res和MOD-res
        int Cipolla(int n) {
            srand(time(NULL));
            if (qp(n, MOD >> 1) == MOD - 1) return -1;
            ll t = mul(rand(), rand());
            while (qp((mul(t, t) - n) % MOD + MOD, MOD >> 1) == 1) t = 1ll * rand() * rand() % MOD;//找到非二次剩余的数,期望循环次数为2
            sqr_i = ((mul(t, t) - n) % MOD + MOD) % MOD;
            int res = qp(spc_Cp(t, 1), MOD + 1 >> 1).x;
            //return res;//返回任何一个解
            return min(res, MOD - res);//返回较小解
        }
}
#endif

#ifdef DF_LP
    //光速幂,处理同底数同模数的幂
    namespace LP {
        //s=sqrt(P)+1
        //x^n=x^{n % s}*x^{n/s*s}
        ll getphi(ll x) {
            ll res = x;
            for (int i = 2; i * i <= x; i++) {
                if (x % i == 0) {
                    res -= res / i;
                    while (x % i == 0) x /= i;
                }
            }
            if (x > 1) res -= res / x;
            return res;
        }
        int base1[N], basesqrt[N];
        int Block_len;
        int Phi;
        ll maxn = 1e10;//模数的最大值
        void init(int x) {//初始化底数为x
            Phi = getphi(MOD);
            Block_len = sqrt(maxn) + 1;
            base1[0] = 1;for (int i = 1;i <= Block_len;i++) base1[i] = 1ll * base1[i - 1] * x % MOD;
            basesqrt[0] = 1;for (int i = 1;i <= Block_len;i++) basesqrt[i] = 1ll * basesqrt[i - 1] * base1[Block_len] % MOD;
        }
        int qp(ull x) {
            x %= Phi;
            return 1ll * basesqrt[x / Block_len] * base1[x % Block_len] % MOD;
        }
    }
#endif

#ifdef DF_binom
    //逆元及组合数
    namespace binom {
        int fact[N], infact[N];//阶乘,阶乘逆元
        int inv(int x, int mod = MOD) { return qp(x, mod - 2, mod); } //费马小定理求逆元
        int __inv(int x, int mod = MOD) { return 1ll * fact[x - 1] * infact[x] % mod; } //预处理求逆元
        void init(int n, int mod = MOD) {
            fact[0] = infact[0] = 1;
            for (int i = 1;i <= n && i < mod;i++) {
                fact[i] = 1ll * fact[i - 1] * i % mod;
                //infact[i] = infact[i - 1] * inv(i,mod) % mod;
            }
            if (mod <= n) {
                infact[mod - 1] = inv(fact[mod - 1], mod);for (int i = mod - 1;i >= 1;i--) infact[i - 1] = 1ll * infact[i] * i % mod;
            }
            else {
                infact[n] = inv(fact[n], mod);for (int i = n;i >= 1;i--) infact[i - 1] = 1ll * infact[i] * i % mod;
            }
        }
        int comb(int a, int b, int mod = MOD) {
            if (b < 0 || a < 0 || b > a) return 0;
            return 1ll * fact[a] * infact[b] % mod * infact[a - b] % mod;
        }
        int invcomb(int a, int b, int mod = MOD) {
            if (b < 0 || a < 0 || b > a) return 0;
            return 1ll * infact[a] * fact[b] % mod * fact[a - b] % mod;
        }
        int __comb(int a, int b, int mod = MOD) {//C(a,b)=A(a,b)/b!
            if (b < 0 || a < 0 || b > a) return 0;
            int res = 1;
            for (int i = a;i >= a - b + 1;i--) res = 1ll * res * i % mod;
            res = 1ll * res * infact[b] % mod;
            return res;
        }
        int lucas(int a, int b, int mod = MOD) {
            if (a < mod && b < mod) return comb(a, b);
            return comb(a % mod, b % mod) * lucas(a / mod, b / mod) % mod;
        }
    }
#endif

#ifdef DF_C
    //递推组合数
    namespace C {
        int c[110][110];//组合数
        void init() {
            c[0][0] = 1;
            for (int i = 1;i <= 100;i++) {
                c[i][0] = 1;
                for (int j = 1;j <= 100;j++) {
                    c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
                }
            }
        }
    }
#endif

#ifdef DF_Stirling
    //递推斯特林数
    namespace Stirling {
        int SA[5010][5010], SC[5010][5010];//两类斯特林数
        void init(int mod = MOD) {
            SA[0][0] = 1;SC[0][0] = 1;
            for (int i = 1;i < 5010;i++) {
                for (int j = 1;j <= i;j++) {
                    SA[i][j] = (SA[i - 1][j - 1] + 1ll * (i - 1) * SA[i - 1][j] % mod) % mod;
                    SC[i][j] = (SC[i - 1][j - 1] + 1ll * j * SC[i - 1][j] % mod) % mod;
                }
            }
        }
    }
#endif

#ifdef DF_NT
    //数论
    namespace NT {

        ll getphi(ll x) {//phi函数
            ll res = x;
            for (int i = 2; i * i <= x; i++) {
                if (x % i == 0) {
                    res -= res / i;
                    while (x % i == 0) x /= i;
                }
            }
            if (x > 1) res -= res / x;
            return res;
        }
        int euler_qp(string n, int mod) {//欧拉降幂 a^n % mod
            //当gcd(b, mod) = 1, a^ b% mod = a ^ (b % phi(mod)) % mod, 从此可以看出当mod为质数时, a^ b% mod = a ^ (b % (mod - 1))
            //gcd(b, mod) != 1, 如果b < phi(mod), a^ b = a ^ b; b >= phi(mod), a^ b = a ^ (b % phi(mod) + phi(mod))
            ll nn = 0;
            int ok = 0;
            ll phi_mod = getphi(mod);
            for (auto i : n) {
                nn = nn * 10 + i - '0';
                if (nn >= phi_mod) {
                    ok = 1;
                    nn %= phi_mod;
                }
            }
            if (ok) nn += phi_mod;
            return nn;
        }

        int mu[N];int phi[N];//mu和phi函数
        int np[N];vector<int> p;//不是质数,存储质数

        void init(int n, int mod = MOD) {//欧拉筛积性函数
            np[0] = np[1] = 1;mu[1] = 1;phi[1] = 1;
            for (int i = 2;i <= n;i++) {
                if (!np[i]) {
                    p.push_back(i);
                    mu[i] = -1;phi[i] = i - 1;
                }
                for (auto j : p) {
                    if (i * j <= n) {
                        np[i * j] = 1;
                        if (i % j == 0) {
                            phi[i * j] = phi[i] * j;
                            break;
                        }
                        else {
                            mu[i * j] = -mu[i];
                            phi[i * j] = phi[i] * (j - 1);
                        }
                    }
                    else {
                        break;
                    }
                }
            }
        }
    }
#endif


#ifdef DF_Primes
    //质数
    namespace Primes {

        int isqrt(ll n) { return sqrtl(n); }
        //返回[1,n]的质数数量
        ll prime_pi(const ll N) {
            if (N <= 1) return 0;
            if (N == 2) return 1;
            const int v = isqrt(N);
            int s = (v + 1) / 2;
            vector<int> smalls(s); for (int i = 1; i < s; ++i) smalls[i] = i;
            vector<int> roughs(s); for (int i = 0; i < s; ++i) roughs[i] = 2 * i + 1;
            vector<ll> larges(s); for (int i = 0; i < s; ++i) larges[i] = (N / (2 * i + 1) - 1) / 2;
            vector<bool> skip(v + 1);
            const auto divide = [](ll n, ll d) -> int { return double(n) / d; };
            const auto half = [](int n) -> int { return (n - 1) >> 1; };
            int pc = 0;
            for (int p = 3; p <= v; p += 2) if (!skip[p]) {
                int q = p * p;
                if ((ll)(q)*q > N) break;
                skip[p] = true;
                for (int i = q; i <= v; i += 2 * p) skip[i] = true;
                int ns = 0;
                for (int k = 0; k < s; ++k) {
                    int i = roughs[k];
                    if (skip[i]) continue;
                    ll d = (ll)(i)*p;
                    larges[ns] = larges[k] - (d <= v ? larges[smalls[d >> 1] - pc] : smalls[half(divide(N, d))]) + pc;
                    roughs[ns++] = i;
                }
                s = ns;
                for (int i = half(v), j = ((v / p) - 1) | 1; j >= p; j -= 2) {
                    int c = smalls[j >> 1] - pc;
                    for (int e = (j * p) >> 1; i >= e; --i) smalls[i] -= c;
                }
                ++pc;
            }
            larges[0] += (ll)(s + 2 * (pc - 1)) * (s - 1) / 2;
            for (int k = 1; k < s; ++k) larges[0] -= larges[k];
            for (int l = 1; l < s; ++l) {
                int q = roughs[l];
                ll M = N / q;
                int e = smalls[half(M / q)] - pc;
                if (e < l + 1) break;
                ll t = 0;
                for (int k = l + 1; k <= e; ++k) t += smalls[half(divide(M, roughs[k]))];
                larges[0] += t - (ll)(e - l) * (pc + l - 1);
            }
            return larges[0] + 1;
        }

        int minp[N];vector<int> p;//最小质因子,存储质数
        void init(int n) {//预处理[1,n]范围质数和最小质因子
            for (int i = 2;i <= n;i++) {
                if (!minp[i]) {
                    p.push_back(i);
                    minp[i] = i;
                }
                for (auto j : p) {
                    if (i * j > n) break;
                    minp[i * j] = j;
                    if (i % j == 0) break;
                }
            }
        }
        int isP(ll x) {
            if (x == 1) return 0;
            for (int i = 2;i <= x / i;i++) {
                if (x % i == 0) return 0;
            }
            return 1;
        }

    }
#endif

}
//init,对应的名字对准!!!
using namespace MATH;

```

### FWT

```c++
namespace FWT {
    struct Poly {
        vector<int> p;
        Poly(int n) { p.resize(n); }
        Poly() {}
        int operator[](const int& k)const { return p[k]; }
        int& operator[](const int& k) { return p[k]; }
        void resize(int n) { p.resize(n); }
        int size() { return p.size(); }
        int deg() const { return p.size() - 1; }
        void clear() { vector<int>().swap(p); }
        void FMT_OR(int lim) {
            for (int mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        p[j + k + mid] = (p[j + k + mid] + p[j + k]) % MOD;
                    }
                }
            }
        }
        void IFMT_OR(int lim) {
            for (int mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        p[j + k + mid] = (p[j + k + mid] - p[j + k] + MOD) % MOD;
                    }
                }
            }
        }
        void FMT_AND(int lim) {
            for (int mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        p[j + k] = (p[j + k] + p[j + k + mid]) % MOD;
                    }
                }
            }
        }
        void IFMT_AND(int lim) {
            for (int mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        p[j + k] = (p[j + k] - p[j + k + mid] + MOD) % MOD;
                    }
                }
            }
        }
        void FWT_XOR(int lim) {
            for (int tmp, mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        tmp = p[j + k + mid];
                        p[j + k + mid] = (p[j + k] - p[j + k + mid] + MOD) % MOD;
                        p[j + k] = (p[j + k] + tmp) % MOD;
                    }
                }
            }
        }
        void IFWT_XOR(int lim) {
            static const int inv2 = (MOD + 1) >> 1;
            for (int mid = 1; mid < lim; mid <<= 1) {
                for (int j = 0; j < lim; j += (mid << 1)) {
                    for (int k = 0;k < mid;k++) {
                        int tmp = p[j + k + mid];
                        p[j + k + mid] = 1ll * inv2 * (p[j + k] - p[j + k + mid] + MOD) % MOD;
                        p[j + k] = 1ll * inv2 * (p[j + k] + tmp) % MOD;
                    }
                }
            }
        }
        friend Poly OR(Poly F, Poly G) {
            int lim = max(F.size(), G.size());
            F.FMT_OR(lim), G.FMT_OR(lim);
            for (int i = 0;i < lim;i++) F[i] = 1ll * F[i] * G[i] % MOD;
            F.IFMT_OR(lim);
            return F;
        }
        friend Poly AND(Poly F, Poly G) {
            int lim = max(F.size(), G.size());
            F.FMT_AND(lim), G.FMT_AND(lim);
            for (int i = 0;i < lim;i++) F[i] = 1ll * F[i] * G[i] % MOD;
            F.IFMT_AND(lim);
            return F;
        }
        friend Poly XOR(Poly F, Poly G) {
            int lim = max(F.size(), G.size());
            F.FWT_XOR(lim), G.FWT_XOR(lim);
            for (int i = 0;i < lim;i++) F[i] = 1ll * F[i] * G[i] % MOD;
            F.IFWT_XOR(lim);
            return F;
        }
    };
}
```



### 求二元一次不定方程

给定$a,b,c$ ，求$ax+by=c$中$x$和$y$的整数解。

首先用exgcd求出一组特解$x_0,y_0$，然后有$ax_0+by_0=gcd(a,b)=d$，如果$c \;mod\;d \neq 0$则无解。

令$base=\frac{c}{d}$，则有$a(x_0\cdot base+\frac{b}{gcd(a,b)})+b(y_0\cdot base-\frac{a}{gcd(a,b)})=c$ 。则有通解为$\begin{cases} x=x_0 \cdot \frac{c}{d} +k\cdot\frac{b}{gcd(a,b)}\\y=y_0\cdot \frac{c}{d}-k\cdot\frac{a}{gcd(a,b)} \end{cases}$ 。则很显然只要对$x_0\cdot \frac{c}{d}$关于$\frac{b}{gcd(a,b)}$取模就能得到$x$的最小非负整数解，对$y$同理，只要对$y_0\cdot \frac{c}{d}$关于$\frac{a}{gcd(a,b)}$取模就能得到$y$的最小非负整数解

### 组合数

```cpp
int P;
namespace CNM {
    const int N = 5e5 + 7;
    int c[N][N];
    void init(int n) {
        for (int i = 0;i <= n;i++)
            for (int j = 0;j <= i;j++)
                c[i][j] = 0 < j && j < i ? (c[i - 1][j - 1] + c[i - 1][j]) % P : 1;
    }
    int C(int n, int m) {
        if (n == m && m == -1) return 1; //* 隔板法特判
        if (n < m || m < 0) return 0;
        return c[n][m];
    }
}
```





#### 扩展中国剩余定理

```c++
//求解线性同余方程组 x%m1=r1,x%m2=r2.... 求解最小非负整数解
ll norm(ll x, int mod = MOD) { return (x % mod + mod) % mod; }
int m[N], r[N];//m为模数,r为余数
auto excrt = [&](int n)->ll {//n为方程组个数
    int m1, m2, r1, r2, k1, k2;
    m1 = m[1], r1 = r[1];
    for (int i = 2;i <= n;i++) {
        m2 = m[i], r2 = r[i];
        int d = exgcd(m1, m2, k1, k2);
        int cm = m1 * m2 / d;
        if ((r2 - r1) % d) return -1;
        k1 *= (r2 - r1) / d;k1 = norm(k1, m2 / d);
        r1 = k1 * m1 + r1;m1 = cm;
    }
    return norm(r1, m1);
    };
```



#### Miller Rabin & Pollard Rho (大数判质数,大数质因子分解)

```c++
srand(time(0));
vector<ll> res;
inline ll qp(ll a, ll n, ll mod) {
    ll res = 1;
    while (n) {
        if (n & 1) res = (__int128)1 * res * a % mod;
        a = (__int128)1 * a * a % mod;
        n = n >> 1ll;
    }
    return res;
}
int base[] = { 0,2,3,5,7,11,13,17,19,23,29,31,37 };
inline bool test(ll n, ll a, ll b, ll x) {
    ll v = qp(x, a, n);
    if (v == 1) return 1;
    int j = 1;
    while (j <= b) {
        if (v == n - 1) break;
        v = (__int128)1 * v * v % n;
        j++;
    }
    if (j > b) return 0;
    return 1;
}
inline bool MR(ll n) {
    if (n < 3 || n % 2 == 0) return n == 2;
    if (n > 37) {
        ll a = n - 1, b = 0;
        while (a % 2 == 0) a >>= 1, b++;
        for (int i = 1; i <= 12; i++) if (!test(n, a, b, base[i])) return 0;
        return 1;
    }
    else {
        for (int i = 1; i <= 12; i++) if (n == base[i]) return 1;
        return 0;
    }
}
inline ll f(ll x, ll c, ll mod) { return ((__int128)1 * x * x % mod + c) % mod; }
inline ll PR(ll n) {
    if (n == 4) return 2;
    std::uniform_int_distribution<ll> Rand(3, n - 1);
    ll x = Rand(mrand), y = x, c = Rand(mrand);
    x = f(x, c, n), y = f(f(y, c, n), c, n);
    for (int lim = 1; x != y; lim = min(lim << 1, 128ll)) {
        ll cnt = 1;
        for (int i = 0; i < lim; i++) {
            cnt = (__int128)1 * cnt * abs(x - y) % n;
            if (!cnt) break;
            x = f(x, c, n), y = f(f(y, c, n), c, n);
        }
        ll d = gcd(cnt, n);
        if (d != 1) return d;
    }
    return n;
}
inline void find(ll x) {
    if (x == 1) return;
    if (MR(x)) { res.push_back(x); return; }
    ll p = x;
    while (p == x) p = PR(x);
    while (x % p == 0) x /= p;
    find(p);find(x);
}
inline void Prime_factor(int x) {
    res.resize(0);
    find(x);
    //sort(res.begin(), res.end());
}
```





### 组合数学

#### 恒等式

##### 递推式

$C_{n}^{m}=C_{n-1}^{m}+C_{n-1}^{m-1}$

##### 分离式

$C_{n}^{x}C_{x}^{y}=C_{n}^{y}C_{n-y}^{x-y}$

##### 吸收式

$C_{n}^{m}=\frac{n}{m}C_{n-1}^{m-1}$

##### 斯特林子集数递推

$SC_{n}^{m}=SC_{n-1}^{m-1}+mSC_{n-1}^{m}$ 

##### 斯特林转换数递推

$SA_{n}^{m}=SA_{n-1}^{m-1}+(n-1)SA_{n-1}^{m}$

##### 平行求和

$\sum\limits_{i=0}^{n}C_{r+i}^{i}=C_{r+n+1}^{n}$

$\sum\limits_{i=0}^{n}C_{r+i}^{i}=C_{r}^{0}+C_{r+1}^{1}+C_{r+2}^{2}+...+C_{r+n}^{n}$

$=C_{r+1}^{0}+C_{r+1}^{1}+C_{r+2}^{2}+...+C_{r+n}^{n}$ （$C_{r}^{0}=C_{r+1}^{0}$）

$=C_{r+2}^{1}+C_{r+2}^{2}+...+C_{r+n}^{n}=C_{r+n+1}^{n}$ 

##### 上指标求和

$\sum\limits_{i=0}^{n}C_{i}^{m}=C_{n+1}^{m+1}$  

$\sum\limits_{i=0}^{n}C_{i}^{m}=C_{m}^{m}+C_{m+1}^{m}+...+C_{n}^{m}$ (忽略为0的项) 

$=C_{m+1}^{m+1}+C_{m+1}^{m}+...+C_{n}^{m}$ （$C_{m}^{m}=C_{m+1}^{m+1}$)

$=C_{m+2}^{m+1}+C_{m+2}^{m}+...+C_{n}^{m}=C_{n+1}^{m+1}$ 

##### 范德蒙德卷积

$\sum\limits_{i=0}^{n}C_{n}^{i}C_{m}^{k-i}=C_{n+m}^{k}$

令$F(x)=(1+x)^{n+m}=(1+x)^n(1+x)^m$ ，按照多项式把两边的式子提取系数

$F[k]=C_{n+m}^{k}=\sum\limits_{i=0}^{n}C_{n}^{i}C_{m}^{k-i}$

推论:

$\sum\limits_{i=-r}^{s}C_{n}^{r+i}C_{m}^{s-i}=C_{n+m}^{s+r}$

令$i=i-r$ ，有$\sum\limits_{i=0}^{s+r}C_{n}^{i}C_{m}^{s-i+r}=C_{n+m}^{s+r}$

##### 将相乘化为相加

方便进行卷积

$ij=C_{i+j}^{2}-C_{i}^{2}-C_{j}^{2}$



##### 贝尔数

将$n$个元素划分为任意个非空子集的方案数

可以发现这是一行斯特林子集数的和，即$\sum\limits_{i=1}^{n}SC_{n}^{i}$，但是这样算比较慢

考虑更快的做法

元素之间有标号，考虑单个集合的EGF，$F(x)=\sum\limits_{i=1}\frac{x^i}{i!}=e^x-1$

集合之间无序，那么就是无序组合背包，$G(x)=\sum\limits_{i=0}\frac{F(x)^i}{i!}=e^{F(x)}=e^{e^x-1}$

模板题：[P5748 集合划分计数](https://www.luogu.com.cn/problem/P5748)

```c++
void Solve(int TIME) {
 
    Poly_init();
    int t;cin >> t;
    Poly f(1e5 + 1);for (int i = 1;i <= 1e5;i++) f.p[i] = infact[i];
    f = Exp(f);
    while (t--) {
        int n;cin >> n;
        cout << fact[n] * f.p[n] % MOD << endl;
    }
 
}
```





#### 斯特林反演

$f(n)=\sum\limits_{i=0}^{n}SC_{n}^{i}g(i)\;⟺\;g(n)=\sum\limits_{i=0}^{n}(-1)^{n-i}SA_{n}^{i}f(i)$

$f(n)=\sum\limits_{i=0}^{n}(-1)^{n-i}SC_{n}^{i}g(i)\;⟺\;g(n)=\sum\limits_{i=0}^{n}SA_{n}^{i}f(i)$

$f(n)=\sum\limits_{i=n}SC_{i}^{n}g(i)\;⟺\;g(n)=\sum\limits_{i=n}(-1)^{i-n}SA_{i}^{n}f(i)$

$f(n)=\sum\limits_{i=n}(-1)^{i-n}SC_{i}^{n}g(i)\;⟺\;g(n)=\sum\limits_{i=n}SA_{i}^{n}f(i)$

#### 二项式反演

二项式反演的四种形式

$f(n)=\sum\limits_{i=0}^{n}(-1)^iC_{n}^{i}g(i)\;⟺\;g(n)=\sum\limits_{i=0}^{n}(-1)^iC_{n}^{i}f(i)$

$f(n)=\sum\limits_{i=0}^{n}C_{n}^{i}g(i)\;⟺\;g(n)=\sum\limits_{i=0}^{n}(-1)^{n-i}C_{n}^{i}f(i)$

$f(n)=\sum\limits_{i=n}(-1)^iC_{i}^{n}g(i)\;⟺\;g(n)=\sum\limits_{i=n}(-1)^{i}C_{i}^{n}f(i)$

$f(n)=\sum\limits_{i=n}C_{i}^{n}g(i)\;⟺\;g(n)=\sum\limits_{i=n}(-1)^{i-n}C_{i}^{n}f(i)$

用生成函数随便证明一下第二个（其他三个也是基本一样的）：

$f[n]=\sum\limits_{i=0}^{n}C_{n}^{i}g[i]=\sum\limits_{i=0}^{n}\frac{n!}{(n-i)!i!}g[i]$

$\frac{f[n]}{n!}=\sum\limits_{i=0}^{n}\frac{1}{(n-i)!}\frac{g[i]}{i!}$

$F[n]=\sum\limits_{i=0}^{n}\frac{1}{(n-i)!}G[i]$  (令$G[i]=\frac{g[i]}{i}$，$F[i]=\frac{f[i]}{i}$)

发现这是一个卷积，并且由于$\sum\limits_{i=0}\frac{x^i}{i!}=e^x$

于是$F=e^x*G $ ，那么$G=e^{-x}*F$

那么$\frac{g[n]}{n!}=\sum\limits_{i=0}^{n}\frac{(-1)^{n-i}}{(n-i)!}\frac{f[i]}{i!}$

那么$g[n]=\sum\limits_{i=0}^{n}\frac{(-1)^{n-i}n!}{(n-i)!i!}f[i]=\sum\limits_{i=0}^{n}(-1)^{n-i}C_{n}^{i}f[i]$

 

 

共有$n$件物品，设$f(x)$表示钦定x个物品非法，其余物品任意的方案数 ， $g(x)$表示刚好x个物品非法，其余物品合法的方案数

由于这个状态设计，$f(x)$往往是好算的。又因为$f(x)=\sum\limits_{i=x}^{n}C_{i}^{x}g(i)$ ，则通过反演有 $g(x)=\sum\limits_{i=x}^{n}(-1)^{i-x}C_{i}^{x}f(i)$

算出$f(x)$后即可通过该式得出$g(x)$

 

#### 生成函数的k阶前缀和与差分

只需要卷上全1序列即$\frac{1}{1-x}$即可做一次前缀和。求$f(x)$的k阶前缀和，即$f(x)(1-x)^{-k}$

只需要卷上${1-x}$即可做一次差分。求$f(x)$的k阶差分，即$f(x)(1-x)^{k}$



####  数的拆分

##### 分拆数(正整数可重复无序拆分)

模板题：[LOJ6268. 分拆数](https://loj.ac/p/6268)

将正整数$n$拆成几个正整数的和。拆法是与顺序无关的。 比如$5=1+4$和$5=4+1$，是同一种拆分方案。

由于拆分是无序的，我们将数字从小到大排列，表示成$n=\sum\limits_{i=1}^{}cnt_{i}i$，其中$cnt_i$表示数字i出现的次数

$F(x)=(1+x+x^2+...)(1+x^2+(x^2)^2+...)(1+x^3+(x^3)^2+...)...=\frac{1}{1-x}\frac{1}{1-x^2}\frac{1}{1-x^3}..=\prod\limits_{i=1}\frac{1}{1-x^i}$

$\prod\limits_{i=1}\frac{1}{1-x^i}=e^{\sum_\limits{i=1}ln(\frac{1}{1-x^i})}$ （通过$ln$和$exp$）

$=e^{\sum_\limits{i=1}\sum_\limits{j=1}\frac{x^{ij}}{j}}$（由$ln(\frac{1}{1-x})=\frac{x}{1}+\frac{x^2}{2}+\frac{x^3}{3}+...$，有$ln(\frac{1}{1-x^i})=\frac{x^i}{1}+\frac{(x^i)^2}{2}+\frac{(x^i)^3}{3}+...$）

指数上这个求和式的复杂度是$\frac{n}{1}+\frac{n}{2}+...+\frac{n}{n}=n(\frac{1}{1}+\frac{1}{2}+...+\frac{1}{n})=nlogn$

多项式$exp$复杂度$nlogn$ 。总时间复杂度$nlogn$。

```c++
 void Solve(int TIME) {
 
    Poly_init();
    int n;cin >> n;
    Poly f(n + 1);
    for (int i = 1;i <= n;i++) {
        for (int j = 1;j * i <= n;j++) {
            f.p[i * j] += qp(j, MOD - 2);
        }
    }
    
    f = Exp(f);
    for (int i = 1;i <= n;i++) cout << f.p[i] << endl;
 
}
```

扩展：如果每个数字$i$有$A[i]$种颜色(保证$A[0]=0$)，并且每个数字的每个颜色都可以无限次使用，并且按字典序排序后，对于两个划分方案只要任意一个位置的颜色或数字不同，就算作不同的方案。

$ans=\prod\limits_{i=1}(\frac{1}{1-x^i})^{A[i]}$

这其实是个$MSET$构造。如果用$\mathcal A$表示组合类，那么元素大小$|x|$定义为拆分出的数字x。$A[x]$表示数字为x的颜色数量。

 

##### 正整数可重复有序拆分

由于拆分是有序的，考虑每个位置的数是什么。

即$(x+x^2+...)(x+x^2+...)..$ (不能是0，所以去掉了$x^0$这一项)

枚举拆分出的数字的数量$k$，设总方案数的生成函数$ans(x)$

$ans(x)=\sum\limits_{k=0}\underbrace{(x+x^2+...)(x+x^2+...)...}_{共k项}=\sum\limits_{k=0}\prod\limits_{i=1}^{k}(x+x^2+...)$

$=\sum\limits_{k=0}(x+x^2+...)^k=\sum\limits_{k=0}(x+x^2+...)^k=\sum\limits_{k=0}(\frac{x}{1-x})^k=\frac{1}{1-\frac{x}{1-x}}$

$=\frac{1-x}{1-2x}=\frac{1}{2}+\frac{1}{2}\cdot \frac{1}{1-2x}$ ，而$\frac{1}{1-2x}=1+2x+(2x)^2+...$ ，则$F[n]=2^n\frac{1}{2}=2^{n-1}$

还有一种证明方法是: 利用隔板法，枚举分成$k$个数，$ans_k=C_{n-1}^{k-1}$即是$x_1+x_2+..+x_k=n$的正整数解的组数

$ans=\sum\limits_{k=1}^{n}C_{n-1}^{k-1}=2^{n-1}$

 

扩展：如果每个数字$i$有$A[i]$种颜色(保证$A[0]=0$)，并且每个数字的每个颜色都可以无限次使用，并且对于两个划分方案只要任意一个位置的颜色或数字不同，就算作不同的方案。

那么$A(x)=\sum\limits_{i=1}A[i]x^i+A[0]x^0=\sum\limits_{i=0}A[i]x^i$，

$ans(x)=\sum\limits_{k=0}A(x)^k=\frac{1}{1-A(x)}$  (类比$\sum\limits_{i=0}x^i=\frac{1}{1-x}$)

这其实是一个$SEQ$构造。如果用$\mathcal A$表示组合类，那么元素大小$|x|$定义为拆分出的数字x。$A[x]$表示数字为x的颜色数量。

##### 不重复无序拆分

考虑每个数，只有$1$个或$0$个的情况。生成函数$F(x)=(1+x^1)(1+x^2)(1+x^3)...$ 

##### 不重复有序拆分

只能想到指数级别的做法...状压然后枚举，如果和刚好为$n$，就将$1$的数量的阶乘计入答案

##### 将$n$划分成多个不大于$m$的数，可重复，无序

考虑每个不大于m的数的数量

$F(x)=(1+x+x^2+...)(1+x^2+(x^2)^2+...)...(1+x^m+(x^m)^2+...)=\prod\limits_{i=1}^{m}\frac{1}{1-x^i}$

dp方法：

$f(n, m)$表示整数$n$的划分中, 每个数都不大于$m$的划分方案.

转移时, 可以分成两种情况

第一种情况 : 划分方案中每个数都小于$m$, 因此方案$f(n, m - 1)$

第二种情况 : 划分方案中至少有一个数为$m$, 那么就在$n$中减去$m$, 得到$f(n - m, m)$

故$f(n, m) = f(n, m - 1) + f(n - m, m)$

##### 将$n$划分成多个不大于$m$的数，不可重复，无序

考虑每个数不大于$m$的数，只有$1$个或$0$个的情况。生成函数$F(x)=\prod\limits_{i=1}^{m}(1+x^i)$ 

dp方法：

$f(n, m) = f(n, m - 1) + f(n - m, m - 1)$





#### 小球入盒

现在有$n$个球，求放入$m$个箱子的方案数（球必须全部放入箱子中）

模板题：[P5824 十二重计数法](https://www.luogu.com.cn/problem/P5824)



```c++
 void Solve(int TIME) {
 
    Poly_init();
    int n, m;cin >> n >> m;
    Poly g(n + 1);for (int i = 0;i <= n;i++) g.p[i] = qp(i, n) * infact[i] % MOD;
    Poly h(n + 1);for (int i = 0;i <= n;i++) h.p[i] = ((i & 1) ? MOD - 1 : 1) * infact[i] % MOD;
    Poly f = g * h;
 
    Poly ff(n + 1);
    for (int i = 1;i <= m;i++) {
        for (int j = 1;j * i <= n;j++) {
            (ff.p[i * j] += inv(j)) %= MOD;
        }
    }
    ff = Exp(ff);
 
    cout << qp(m, n) << endl;
    cout << (n > m ? 0 : fact[m] * infact[m - n] % MOD) << endl;
    cout << (m > n ? 0 : f.p[m] * fact[m] % MOD) << endl;
 
    int res4 = 0;for (int i = 0;i <= min(m, n);i++) res4 = (res4 + f.p[i]) % MOD;cout << res4 << endl;
    cout << (n > m ? 0 : 1) << endl;
    cout << (m > n ? 0 : f.p[m]) << endl;
 
    cout << comb(n + m - 1, m - 1) << endl;
    cout << (n > m ? 0 : comb(m, n)) << endl;
    cout << comb(n - 1, m - 1) << endl;
 
 
    cout << ff.p[n] << endl;
    cout << (n > m ? 0 : 1) << endl;
    cout << (m > n ? 0 : ff.p[n - m]) << endl;
 
}
```



大致框架：

球相同时，可以看作是数的划分，然后看盒，如果盒相同便是无序划分，盒不同便是有序划分；

球不同时，如果盒相同，则是斯特林子集数，盒不同便是斯特林子集数算上盒子的顺序。当然斯特林子集数一个重要条件就是无空箱，这里需要额外再作讨论。

##### 球同，箱不同，允许空箱

$x_1+x_2+...+x_m=n$的非负整数解的个数 。考虑先给所有盒子放上$1$个球。

然后隔板法。$C_{n+m-1}^{m-1}$

##### 球同，箱不同，无空箱

即$x_1+x_2+...+x_m=n$的正整数解的个数 

隔板法。 $C_{n-1}^{m-1}$

##### 球同，箱不同，箱内至多一个球

当$n>m$，方案数为0，因为每个球必须放入箱子中

当$n\leq m$，为$C_{m}^{n}$，即选择哪几个箱子放入球

##### 球同，箱同，允许空箱

将n无序的拆成m个正整数的方案数。

由于盒子相同，考虑枚举$i$，拥有$i$个球的盒子的个数。显然$i_{max}=m$ 。

$F(x)=(1+x^0+(x^0)^2+...)(1+x+x^2+...)(1+x^2+(x^2)^2+...)..$ .

$=m\prod\limits_{i=1}^{m}(1+x^i+(x^i)^2+..)=\prod\limits_{i=1}^{m}\frac{m}{(1-x^i)}$ ， $res=F[n]$ 。 由于$m$对于答案无影响，可以直接略去。

$F(x)=e^{ln\prod\limits_{i=1}^{m}\frac{1}{1-x^i}}=e^{\sum\limits_{i=1}^{m}ln\frac{1}{1-x^i}}=e^{\sum\limits_{i=1}^{m}\sum\limits_{j=1}\frac{x^{ij}}{j}}$

$res=[x^n]\prod\limits_{i=1}^{m}\frac{1}{(1-x^i)}=[x^n]e^{\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{\lfloor\frac{n}{i} \rfloor}\frac{x^{ij}}{j}}$

##### 球同，箱同，无空箱

将$n$无序的拆成$m$个非负整数的方案数。

先给每个箱子放上一个球，再进行上述的操作。

$res=[x^{n-m}]\prod\limits_{i=1}^{m}\frac{1}{(1-x^i)}$

##### 球同，箱同，箱内至多一个球

当$n>m$，方案数为$0$，因为每个球必须放入箱子中

当$n\leq m$，为$1$。 即$[n\leq m]$

##### 球不同，箱不同，允许空箱

每个球有m种放法。$res=m^n$

##### 球不同，箱不同，无空箱

多了有序性的斯特林子集数。$m!SC_{n}^{m}$

##### 球不同，箱不同，箱内至多一个球

当$n>m$，方案数为$0$，因为每个球必须放入箱子中

当$n\leq m$，$m(m-1)...(m-n+1)=m^{\underline{n}}$

##### 球不同，箱同，允许空箱

枚举$i$个盒子非空。$res=\sum\limits_{i=1}^{m}SC_{n}^{i}$

##### 球不同，箱同，无空箱

斯特林子集数。$SC_{n}^{m}$

##### 球不同，箱同，箱内至多一个球

当$n>m$，方案数为$0$，因为每个球必须放入箱子中

当$n\leq m$，为$1$。 即$[n\leq m]$



#### 无标号构造（即$A(x)$无标号，$A(x)$之间相同）

##### $Sequence$构造（序列构造）

有序排列背包。

对于组合类$\mathcal A$，$SEQ(\mathcal A)$表示生成不定长序列的构造

$SEQ(\mathcal A)=\sum\limits_{k=0}\mathcal A^{k}$，规定$A[0]=0$

写成生成函数：$\mathcal B=SEQ(\mathcal A)\;⇒\;B(x)=\sum\limits_{k=0}A(x)^k=\frac{1}{1-A(x)}$

举例：$\mathcal C=\{0，1\}$，显然$|0|=|1|=1$（这里$|x|$表示组合对象也就是元素的大小）

记$\mathcal S$表示$01$串，即$\mathcal S=SEQ(\mathcal C)$

$C[x]=2x$ ，因为数量为$1$的元素有两个。 $S[x]=\frac{1}{1-C[x]}=\frac{1}{1-2x}$

 

 

##### $Multiset$构造（多重集构造）

完全背包。

定义$G$为置换群列，$G$的元素中，($x_1$，$x_2$，$x_3$) ，($x_2$，$x_1$，$x_3$) ，($x_3$，$x_1$，$x_2$)等等都视作是相同的组合

$MSET(\mathcal A)=SEQ(\mathcal A)/G_{\mathcal A}$

可以发现$MSET$可以视作是不定长字典序不降序列的构造。

于是按照规定的顺序枚举$\mathcal A$的所有对象， 每次选择加入若干个（可以是$0$个）

$\mathcal B=MSET(\mathcal A)$

$B(x)=\prod\limits_{a\in \mathcal A}(1+x^{|a|}+(x^{|a|})^2+...)=\prod\limits_{a\in \mathcal A}\sum\limits_{k=0}x^{|a|k}=\prod\limits_{a\in \mathcal A}(\frac{1}{1-x^{|a|}})=\prod\limits_{i=1}(\frac{1}{1-x^i})^{A[i]}$

$=e^{ln{\prod\limits_{i=1}(\frac{1}{1-x^i})^{A[i]}}}=e^{{\sum\limits_{i=1}{A[i]}ln(\frac{1}{1-x^i})}}$

$=e^{{\sum\limits_{i=1}{A[i]}\sum\limits_{j=1}\frac{x^{ij}}{j}}}$ （到这一步，已经可以$O(nlogn)$暴力完成求和，再做多项式$exp$即可）

$=e^{{\sum\limits_{j=1}\frac{1}{j}\sum\limits_{i=1}{A[i]}{x^{ij}}}}$ （交换枚举次序）

$=e^{{\sum\limits_{j=1}\frac{1}{j}\sum\limits_{i=0}{A[i]}{(x^{j})^i}}}=e^{{\sum\limits_{j=1}\frac{A(x^j)}{j}}}=e^{{\sum\limits_{i=1}\frac{A(x^i)}{i}}}$

模板题：[P4389 付公主的背包](https://www.luogu.com.cn/problem/P4389)

```c++
void Solve(int TIME) {
 
    Poly_init();
    int n, m;cin >> n >> m;
    vi cnt(m + 1);
    for (int i = 1;i <= n;i++) {
        int x;cin >> x;
        cnt[x]++;
    }
    Poly f(m + 1);
    for (int i = 1;i <= m;i++) {
        for (int j = 1;j * i <= m;j++) {
            (f.p[i * j] += cnt[i] * qp(j, MOD - 2) % MOD) %= MOD;
        }
    }
    f = Exp(f);
    for (int i = 1;i <= m;i++) cout << f.p[i] << endl;
    
}
```



##### $Powerset$构造

01背包。

枚举每个元素是否存在。 即$\mathcal B=PSET(\mathcal A) \;⇒\;B(x)=\prod\limits_{a\in \mathcal A}(1+x^{|a|}) =\prod\limits_{i=0 }(1+x^{i})^{A[i]}$

考虑用$ln$和$Exp$，$B(x)=e^{ln\prod\limits_{i=0}(1+x^i)^{A[i]}}=e^{\sum\limits_{i=0}{A[i]}ln(1+x^i)}=e^{\sum\limits_{i=0}{A[i]}\sum\limits_{j=1}\frac{(-1)^{j-1}x^{ij}}{j}}$

$=e^{\sum\limits_{j=1}\frac{(-1)^{j-1}}{j}\sum\limits_{i=0}{A[i]x^{ij}}}=e^{\sum\limits_{j=1}\frac{(-1)^{j-1}A(x^j)}{j}}$

指数上的部分，暴力$nlogn$即可。

 

#### 有标号构造（即$A(x)$有标号，$A(x)$之间不同）

##### 有标号$Sequence$构造

同无标号$Sequence$构造。把$OGF$改成$EGF$即可。

##### $Set$构造（$Exp$的组合意义）

无序组合背包。

$\mathcal B=SET(\mathcal A) \;⇒\;B(x)=\sum\limits_{i=0}\frac{A(x)^i}{i!}=e^{A(x)}$

把$A(x)$视作一个整体，可以发现$B(x)$是若干个$A(x)$生成的不定长无序集合（除以$i!$的意义就是将有序变成无序）。

举例：无向联通图就是$A(x)$，无向图就是$e^{A(x)}$ ,因为无向图是由若干个无向联通图无序组合而成。

可以这样理解，若干个无向连通图之间两两不同（表示$A(x)$之间有标号），但是有序排列起来还是同一张无向普通题，所以只需要无序组合起来。

 



#### 狄利克雷卷积

**下取整的性质**

$\left\lfloor\frac {\lfloor\frac{a}{b}\rfloor}{c}\right\rfloor=\lfloor \frac{a}{bc} \rfloor$。 证明略。

**上取整的性质**

$\left\lceil\frac {\lceil\frac{a}{b}\rceil}{c}\right\rceil= \lceil \frac{a}{bc} \rceil$

证明：

$\left\lceil\frac {\lceil\frac{a}{b}\rceil}{c}\right\rceil=\left\lfloor\frac {\lfloor\frac{a+b-1}{b}\rfloor+c-1}{c}\right\rfloor=\left\lfloor\frac {\lfloor\frac{a+b-1+bc-b}{b}\rfloor}{c}\right\rfloor=\left\lfloor\frac {a+bc-1}{bc}\right\rfloor= \lceil \frac{a}{bc} \rceil$

**$g=\mu *f$，已知$f$求出$g$** 

这里有三种方法



```c++
 for (int i = 1; i <= n; i++) g[i] = 0;
for (int i = 1; i <= n; i++) {
    for (int j = 1; i * j <= n; j++) {
        g[i * j] = (g[i * j] + mu[i] * f[j]) % mod;
    }
}
// 依照定义，O(nlogn)
```

```c++
 for (int i = 1; i <= N; i++) g[i] = f[i];
for (int i = 1; i <= N; i++) {
    for (int j = 2; i * j <= N; j++) {
        g[i * j] = (g[i * j] - g[i]) % MOD;
    }
}
// 类似求狄利克雷卷积逆的方式,不需要线性筛 mu ,O(nlogn)
```

```c++
 for (int i = 1; i <= n; i++) g[i] = f[i];
for (auto i : p) {
    for (int j = n / i; j >= 1; j--) {
        g[i * j] = (g[i * j] - g[j]) % MOD;
    }
}//O(nloglogn)
```

第三种可以理解为$dp$，设$g(i,n)=\sum\limits_{d|n,d只含前i种质因子}f(d)\mu(\frac{n}{d})$

那么$g(i,n)=\begin{cases} g(i-1,n), \;\;\;p_i\nmid n \\g(i-1,n)-g(i-1,\frac{n}{p_i}),\;\;\;p_i|n \end{cases}$

 

#### 筛法

**线性筛（欧拉筛）**

枚举每个数$x$，筛去它们的$p_i$倍（$p_i$为质数）。当$x\%p_i=0$时，不需要再往下筛直接$break$，因为$p_i$是递增的，所以$x$乘上其他的质数的结果一定会被$p_i$的倍数筛掉。显然这样每个数只会被筛一次，故复杂度为$O(n)$。

任何积性函数都可以进行线性筛，所有线性筛积性函数都是基于线性筛质数的。

若想线性筛出积性函数$f(x)$，就必须快速计算出以下函数值，$f(1),f(p),f(p^k)$，其中$p$为质数。

考虑筛的过程中，$i\cdot p$会被$i$乘上$p$给筛掉，将$i$唯一分解得到$p_1^{c_1}p_2^{c_2}\cdots p_k^{c_k}$，其中$p_1\le p_2\le \cdots\le p_k$

则一定有$p\leq p_1$，这是显然的，因为一旦第一次遇到$i\%p=0$也就是最小质因子时，直接$break$了。

如果$p< p_1$，那么$p$与$i$互质，可以直接得到$f(i\cdot p)=f(i)f(p)$

如果$p=p1$，这是需要对$i$记录一个$low_i$，表示$i$中最小质因子的指数次幂，即$low_i=p_1^{c_1}$

如果$i$除掉$low_i$，那么结果中最小质因子一定大于$p_1$，从而得到$gcd(\frac{i}{low_i},low_i\cdot p)=1$，那么$f(i\cdot p)=f(\frac{i}{low_i})f(low_i\cdot p)$

 

举例

欧拉函数$\varphi$，$\varphi(1)=1$，$\varphi(p)=p-1$。当$i\%p\neq0$时，$i$和$p$互质，$\varphi(i\cdot p)=\varphi(i)\cdot \varphi(p)=\varphi(i)\cdot(p-1)$；当$i\%p=0$时，$\varphi(i\cdot p)=\varphi(\frac{i}{p^c})\varphi(p^{c+1})=\frac{\varphi(i)}{p^c(1-\frac1{p})}p^{c+1}(1-\frac{1}{p})=p\cdot \varphi(i)$

```c++
 void init(int mod = MOD) {
 
    np[0] = np[1] = 1;phi[1] = 1;
    for (int i = 2;i < N;i++) {
        if (!np[i]) {
            p.push_back(i);
            phi[i] = i - 1;
        }
        for (auto j : p) {
            if (i * j < N) {
                np[i * j] = 1;
                if (i % j == 0) {
                    phi[i * j] = phi[i] * j;
                    break;
                }
                else {
                    phi[i * j] = phi[i] * (j - 1);
                }
            }
            else {
                break;
            }
        }
    }
}
```

莫比乌斯函数$\mu$，$\mu(1)=1$，$\mu(p)=-1$。当$i\%p\neq0$时，$i$和$p$互质，$\mu(i\cdot p)=\mu(i)\cdot \mu(p)=-\mu(i)$；当$i\%p=0$时，$\mu(i\cdot p)=\mu(\frac{i}{p^c})\mu(p^{c+1})=\mu(\frac{i}{p^c})\cdot 0=0$

```c++
 void init(int mod = MOD) {
 
    np[0] = np[1] = 1;mu[1] = 1;
    for (int i = 2;i < N;i++) {
        if (!np[i]) {
            p.push_back(i);
            mu[i] = -1;
        }
        for (auto j : p) {
            if (i * j < N) {
                np[i * j] = 1;
                if (i % j == 0) {
                    mu[i] = 0;
                    break;
                }
                else {
                    mu[i * j] = -mu[i];
                }
            }
            else {
                break;
            }
        }
    }
}
```



约数个数函数$d$，$d(1)=1$，$d(p)=2$ 。当$i\%p\neq0$时，$i$和$p$互质，$d(i\cdot p)=d(i)\cdot d(p)=2d(i)$；当$i\%p=0$时，$d(i\cdot p)=d(\frac{i}{p^c})d(p^{c+1})=\frac{d(i)}{c+1}(c+2)$

约数和函数$\sigma$，$\sigma(1)=1$，$\sigma(p)=p+1$，当$i\%p\neq0$时，$i$和$p$互质，$\sigma(i\cdot p)=\sigma(i)\cdot \sigma(p)=(p+1)\sigma(i)$；当$i\%p=0$时，$\sigma(i\cdot p)=\sigma(\frac{i}{p^c})\sigma(p^{c+1})=\frac{\sigma(i)}{\sum\limits_{i=0}^{c}p^i}\sum\limits_{i=0}^{c+1}p^i$

```c++
 void init(int mod = MOD) {
 
    np[0] = np[1] = 1;
    sigma[1] = 1;low[1] = 1;
    for (int i = 2;i < N;i++) {
        if (!np[i]) {
            p.push_back(i);
            sigma[i] = i + 1;low[i] = i;
        }
        for (auto j : p) {
            if (i * j < N) {
                np[i * j] = 1;
                if (i % j == 0) {
                    low[i * j] = low[i] * j;
                    if (i == low[i]) sigma[i * j] = sigma[i] * j + 1;
                    else sigma[i * j] = sigma[i / low[i]] * sigma[low[i] * j];
                    break;
                }
                else {
                    low[i * j] = j;
                    sigma[i * j] = sigma[i] * (j + 1);
                }
            }
            else {
                break;
            }
        }
    }
}
```

下面这个不是积性函数，但是可以用来求解上面的东西，并且也可以线性筛。

最小次幂函数$low$，即$n=p_1^{c_1}p_2^{c_2}...p_k^{c_k}$，$p_1<p_2<...<p_k$，那么$low(n)=p_1^{c_1}$。

当$i\%p\neq0$时，$i$中最小质因子大于$p$，$low(i\cdot p)=p$；当$i\%p=0$时，$i$中最小质因子为$p$，$low(i\cdot p)=p\cdot low(i)$

**杜教筛**

模板题：[P4213 - 杜教筛](https://www.luogu.com.cn/problem/P4213)

在亚线性时间内求出某些特征积性函数的前缀和

所谓某些特征积性函数，指的是存在一个对应的积性函数$g$，使得$f*g=h$，且$g$，$h$的前缀和可以$O(1)$得知的函数$f$ 。

记$F,G,H$分别为$f,g,h$的前缀和，那么$H(n)=\sum\limits_{i=1}^{n}\sum\limits_{d|n}f(\frac{i}{d})g(d)=\sum\limits_{d=1}^{n}g(d)\sum\limits_{j=1}^{\lfloor \frac{n}{d} \rfloor}f(j)$

$=\sum\limits_{d=1}^{n}g(d)F(\lfloor \frac{n}{d} \rfloor)=g(1)F(n)+\sum\limits_{d=2}^{n}g(d)F(\lfloor \frac{n}{d} \rfloor)$

由于$g$为积性函数，那么$g(1)=1$，于是就有$F(n)=H(n)-\sum\limits_{d=2}^{n}g(d)F(\lfloor \frac{n}{d} \rfloor)$

注意到，如果从小到大筛的话，右边的$F$是已知的，而$G,H$都可以$O(1)$求得，则对$\lfloor\frac{n}{d} \rfloor$整除分块，单论复杂度$O(\sqrt{n})$

事实上，我们一般只关心一个点$n$的$F(n)$，故该式子一般递归求解。预处理$n^{\frac{2}{3}}$内的$f$值时，可以取得最优复杂度。

 

举例

求$\mu$的前缀和，注意到$\mu*1=\varepsilon$，于是有$\sum\limits_{i=1}^{n}\mu(i)=\sum\limits_{i=1}^{n}\varepsilon(i)-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}\mu(i)=1-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}\mu(i)$

求$\varphi$的前缀和，注意到$\varphi*1=id$，$\sum\limits_{i=1}^{n}\varphi(i)=\sum\limits_{i=1}^{n}id(i)-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}\varphi(i)=\frac{n(n+1)}{2}-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}\varphi(i)$

求$id\cdot\mu$的前缀和，我们需要找到合适的函数$g$，注意到$id*id=\sum\limits_{d|n}d\frac{n}{d}=\sum\limits_{d|n}n=nd(n)$

不妨令$g=id$尝试一下。于是有$(id\cdot \mu)*id=\sum\limits_{d|n}d\mu(d)\frac{n}{d}=n\sum\limits_{d|n}\mu(d)=n(\mu*1)=n\cdot \varepsilon(n)=\varepsilon(n)$

所以$\sum\limits_{i=1}^{n}id\cdot\mu(i)=\sum\limits_{i=1}^{n}\varepsilon(i)-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}id\cdot\mu(i)=1-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}id\cdot\mu(i)$

求$id\cdot\varphi$的前缀和，我们需要找到合适的函数$g$，与上面等同，$g=id$，于是有$(id\cdot \varphi)*id=\sum\limits_{d|n}d\varphi(d)\frac{n}{d}=n\sum\limits_{d|n}\varphi(d)=n(\varphi*1)=n\cdot id(n)=n^2=id_2(n)$

所以$\sum\limits_{i=1}^{n}id\cdot\varphi(i)=\sum\limits_{i=1}^{n}id_2(i)-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}id\cdot\varphi(i)=\frac{n(n+1)(2n+1)}{6}-\sum\limits_{d=2}^{n}\sum\limits_{i=1}^{\lfloor \frac{n}{d} \rfloor}id\cdot\varphi(i)$



```c++
 int Du_sieve(int n, int pre_f[], map<int, int>& mp) {//f*g=h,求f的前缀和
    if (n < N) return pre_f[n];
    if (mp.count(n)) return mp[n];
    int res = n * (n + 1) / 2;//h=id前缀和
    for (int l = 2, r;l <= n;l = r + 1) {
        r = n;
        if (n / l) r = min(r, n / (n / l));
        res = (res - Du_sieve(n / l, pre_f, mp) * (r - l + 1));//(r-l+1)为g=1前缀和
    }
    return mp[n] = res;
}
```

 

##### 整除分块

对于$\lfloor\frac{n}{i}\rfloor$类似的式子，对于一个左边界$l$，其值为$k=\lfloor\frac{n}{l}\rfloor$，右边界$r$即找满足$k\le \lfloor\frac{n}{i}\rfloor$的最大的$i$ ，也就是使得$ik\le n$最大的$i$，即$r=\lfloor\frac{n}{k}\rfloor$，即$r= \left\lfloor \frac{n}{\lfloor\frac{n}{l} \rfloor} \right \rfloor$

```c++
int lim = n;
for (int l = 1, r;l <= lim;l = r + 1) {
    r = lim;
    if (n / l) r = min(r, n / (n / l));
    //操作
}
```

拓展

**给定$n$，求$\lfloor\frac{n}{ai+b}\rfloor$的整除分块**

先令$i'=ai+b$，那么$l'=al+b$，$r'=ar+b$。 令$k=\lfloor \frac{n}{l'} \rfloor$，$r'$是最大的$i'$使得$ki'\le n$，那么$r'=\lfloor \frac{n}{k} \rfloor$

带入$l'=al+b$，得到$k=\lfloor \frac{n}{al+b} \rfloor$ ，$r'=\left\lfloor \frac{n}{\lfloor \frac{n}{al+b} \rfloor} \right\rfloor$

$r'=ar+b=\left\lfloor \frac{n}{\lfloor \frac{n}{al+b} \rfloor} \right\rfloor$ ，那么$r=\left\lfloor\frac{\left\lfloor \frac{n}{\lfloor \frac{n}{al+b} \rfloor} \right\rfloor-b}{a}\right\rfloor$

**给定$n$，求$\lfloor\frac{n}{i^2}\rfloor$的整除分块**

令$i'=i^2$，那么$l'=l^2$，$r'=r^2$。令$k=\lfloor\frac{n}{l'}\rfloor$，$r'$是最大的$i'$使得$ki'\le n$，那么$r'=\lfloor\frac{n}{k}\rfloor=\left\lfloor\frac{n}{\lfloor\frac{n}{l^2}\rfloor}\right\rfloor$

$r'=r^2=\left\lfloor\frac{n}{\lfloor\frac{n}{l^2}\rfloor}\right\rfloor$ ，得到$r=\left\lfloor\sqrt{\left\lfloor\frac{n}{\lfloor\frac{n}{l^2}\rfloor}\right\rfloor}\right\rfloor$

**给定$n$，求$\lceil\frac{n}{i}\rceil$的整除分块**

由于$\lceil\frac{n}{i}\rceil=\lfloor\frac{n+i-1}{i}\rfloor$ ，令$k=\lfloor\frac{n+l-1}{l}\rfloor$，$r$是最大的$i$使得$ik\le n+i-1$，那么$r=\lfloor\frac{n-1}{k-1} \rfloor=\left\lfloor\frac{n-1}{\lfloor\frac{n+l-1}{l}\rfloor-1}\right \rfloor$

 

#### 积性函数大赏

积性函数：一个定义域为正整数$n$的算术函数$f(n)$，有如下性质：$f(1)=1$，且当$a,b$互质时，$f(ab)=f(a)f(b)$。如果一个数字$n=p_1^{c_1} p_2^{c_2} \cdots p_k^{c_k}$，那么$f(n)=f(p_1^{c_1})f(p_2^{c_2}) \cdots f(p_k^{c_k})$，那么研究一个积性函数$f$可以转化为研究$f(p^c)$

完全积性函数：$f(1)=1$，且对任意两个正整数$a,b$都有$f(ab)=f(a)f(b)$

性质1：两个积性函数的狄利克雷卷积必定是积性函数。

性质2：狄利克雷卷积满足交换律，结合律，分配律。交换律即$f*g=g*f$，也就是$\sum\limits_{d|n}f(d)g(\frac{n}{d})=\sum\limits_{d|n}f(\frac{n}{d})g(d)$。分配律即$(f+g)*h=f*h+g*h$。结合律即$(f*g)*h=f*(g*h)$

**1.**

单位函数：$\varepsilon(n)=[n=1]$ （完全积性）

**2.**

幂函数：$id_k(n)=n^k$  （完全积性）

**3.**

常数函数：幂函数中，$k=0$时， $id_0(n)=1(n)=1$ ，或作$I(n)=1$ （完全积性）

**4.**

恒等函数：幂函数中，$k=1$时， $id_1(n)=id(n)=n$ （完全积性）

**5.**

除数函数：$\sigma_k(n)$，$n$所有正因数的$k$次幂之和。 这里$k$可以是任何复数和实数。

**6.**

因数个数函数：除数函数中，$k=0$时，$\sigma_o(n)=d(n)=\prod\limits_{i=1}^{k}(c_i+1)$ ，$n$的正因数个数，$n=p_1^{c_1}p_2^{c_2}\cdots p_k^{c_k}$

**7.**

因数和函数：除数函数中，$k=1$时，$\sigma_1(n)=\sigma(n)=\prod\limits_{i=1}^{k}\sum\limits_{j=0}^{c_i}p_i^j$ ，$n$的所有正因数之和，$n=p_1^{c_1}p_2^{c_2}\cdots p_k^{c_k}$

**8.**

莫比乌斯函数：$\mu(n)=\begin {cases} 1,n=1\\0,n含有平方因子\\(-1)^k ,n=\prod\limits_{i=1}^{k}p_i,p_i为质数 \end{cases} $

性质：$\sum\limits_{d|n}\mu(d)=\varepsilon(n)$ 即$\mu*1=\varepsilon$ 。这意味着在狄利克雷卷积中，$\mu$是常数函数$1$的逆元。

证明：设$n=\prod\limits_{i=1}^{k}p_i^{c_i}$，$n'=\prod\limits_{i=1}^{k}p_i$ ，那么$\sum\limits_{d|n}\mu(d)=\sum\limits_{d|n'}\mu(d)=\sum\limits_{i=0}^{k}C_{k}^{i}(-1)^i=(1+(-1))^k=[k=0]=[n=1]$

扩展：$[gcd(i,j)=1]=\sum\limits_{d|gcd(i,j)}\mu(d)$ ，将$n$直接用 $ gcd(i,j) $ 替换即可

**9.**

欧拉函数：$\varphi(n)=\sum\limits_{i=1}^{n}[gcd(i,n)=1]$ ，表示$[1,n]$中与$n$互质的数的个数

计算公式： $n=\prod\limits_{i=1}^{k}p_i^{c_i}$，$p_i$为质数 。 $\varphi(n)=n(1-\frac1{p_1})(1-\frac1{p_2})\cdots(1-\frac1{p_k})$

感性的证明：以$12$举例，$12$有两个质因子$2,3$。在$[1,12]$中，有$\frac1{2}$的数是$2$的倍数，所以有$1-\frac1{2}$是数不是$2$的倍数，在这不是$2$的倍数的数中，有$\frac{1}{3}$的数是3的倍数，所以既不是$2$的倍数也不是$3$的倍数的数字有$12(1-\frac1{2})(1-\frac{1}{3})$ 。

性质1：若$p$是质数，$\varphi(p)=p-1$ 。

性质2：若$p$是质数，$n=p^k$，$\varphi(n)=p^k-p^{k-1}$ 。证明：$\varphi(n)=n(1-\frac{1}{p})=p^k(1-\frac1{p})=p^k-p^{k-1}$

性质3：$\sum\limits_{d|n} \varphi(d)=n $即  $\varphi*1=id$ 。

证明：$n=\prod\limits_{i=1}^{k}p_i^{c_i}$，由于$\varphi$为积性函数，故只需要证明当$n'=p^c$时，$\varphi*1=\sum\limits_{d|n'}\varphi(d)1(\frac{n'}{d}) = id$

显然$d=p^0,p^1,p^2,...,p^c$ ，那么$\sum\limits_{i=0}^{c}\varphi(p^i)=\sum\limits_{i=0}^{c}p^i(1-\frac{1}{p})=1+(1-\frac{1}{p})\sum\limits_{i=1}^cp^i=1+(1-\frac{1}{p})\frac{p(1-p^c)}{1-p}=p^c=id(n')$

 

狄利克雷函数与欧拉函数的联系：对$\varphi*1=id$两边同时卷上$\mu$，得到$\varphi*1*\mu=id*\mu$

由于狄利克雷卷积满足结合律，$\varphi*\varepsilon=id*\mu$即$\varphi=\mu*id$

**10.**

$\omega(n)$表示$n$的质因子数目(不可重复)  ，例如$n=p_1^{c_1} p_2^{c_2} \cdots p_k^{c_k}$ ，$\omega(n)=k$

**11.**

$\Omega(n)$表示$n$的质因子数目(可重复) ，例如$n=p_1^{c_1} p_2^{c_2} \cdots p_k^{c_k}$ ，$\Omega(n)=\sum\limits_{i=1}^{k}c_i$





#### 莫比乌斯反演

$f(n)=\sum\limits_{d|n}g(d)$ ，那么有$g(n)=\sum\limits_{d|n}\mu(d)f(\frac{n}{d})$ 。即$f=g*1$ ，那么有$g=\mu*f$

证明：

两边同时乘以$\mu$得到$f*\mu=g*1*\mu=g*\varepsilon=g$

 



### 线性代数

#### 矩阵板子

```c++
template<typename T>
struct mat {
    T m[101][101];
    mat() {
        memset(m, 0, sizeof m);
    }
    mat(int epsilon) {
        memset(m, 0, sizeof m);
        for (int i = 1;i <= 100;i++) m[i][i] = 1;
    }
    T& operator()(int i, int j) { return m[i][j]; }
    T operator()(int i, int j)const { return m[i][j]; }
    int is_I() {
        for (int i = 1;i <= 100;i++) {
            for (int j = 1;j <= 100;j++) {
                if (i == j && m[i][j] != 1) return 0;
                if (i != j && m[i][j] != 0) return 0;
            }
        }
        return 1;
    }
};



//矩阵乘法
template<typename T>
mat<T> operator *(const mat<T>& x, const mat<T>& y) {
    mat<T> t;
    for (int k = 1;k <= 100;k++) {
        for (int i = 1;i <= 100;i++) {
            for (int j = 1;j <= 100;j++) {
                (t.m[i][j] += x.m[i][k] * y.m[k][j] % MOD) %= MOD;
            }
        }
    }
    return t;
}

//矩阵快速幂
mat<int> qp(mat<int> a, int k) {
    mat<int> res;//res = I;
    while (k) {
        if (k & 1) res = res * a;
        a = a * a;
        k >>= 1;
    }
    return res;
}

//光速幂,处理同底数同模数的幂
namespace LP_Mat {
    ll getphi(ll x) {
        ll res = x;
        for (int i = 2; i * i <= x; i++) {
            if (x % i == 0) {
                res -= res / i;
                while (x % i == 0) x /= i;
            }
        }
        if (x > 1) res -= res / x;
        return res;
    }
    mat<int> base1[N], basesqrt[N];
    int Block_len;
    int Phi;
    ll maxn = 1e10;//模数的最大值
    void init(mat<int> x) {//初始化底数为x
        Phi = getphi(MOD);
        Block_len = sqrt(maxn) + 1;
        base1[0] = mat<int>(1);for (int i = 1;i <= Block_len;i++) base1[i] = base1[i - 1] * x;
        basesqrt[0] = mat<int>(1);for (int i = 1;i <= Block_len;i++) basesqrt[i] = basesqrt[i - 1] * base1[Block_len];
    }
    mat<int> qp(ull x) {
        x %= Phi;
        return basesqrt[x / Block_len] * base1[x % Block_len];
    }
}



const ld eps = 1e-9;

//消成上三角矩阵
//n行m列矩阵,其中第m列为增广矩阵(也可以不是增广矩阵,对于单个矩阵,用于求n行m列的矩阵的秩,在r--后面return r即可)
int gauss(mat<ld>& a, int n, int m) {
    int r, c;//行,列
    for (r = 1, c = 1;r <= n && c < m;r++, c++) {
        int t = r;
        for (int i = r + 1;i <= n;i++) //找最大防误差
            if (abs(a.m[i][c]) > abs(a.m[t][c])) t = i;
        if (t != r) swap(a.m[r], a.m[t]);
        if (abs(a.m[r][c]) < eps) {//该列都为0
            r--;continue;
        }
        for (int i = r + 1;i <= n;i++) {
            if (abs(a.m[i][c]) < eps) continue;
            ld tt = a.m[i][c] / a.m[r][c];
            for (int j = c;j <= m;j++) a.m[i][j] -= tt * a.m[r][j];
        }
    }
    r--;
    if (r < m - 1) {
        for (int i = r + 1;i < m;i++) {
            if (abs(a.m[i][c]) > eps) return -1;//无解. 存在0=非0
        }
        return (m - 1) - r;//无穷多解,返回自由元数量.(总共m-1个变量)
    }
    for (int i = m - 1;i >= 1;i--) {
        for (int j = i + 1;j <= m - 1;j++) a.m[i][m] -= a.m[i][j] * a.m[j][m];
        a.m[i][m] /= a.m[i][i];
    }
    return 0;//唯一解
}



//同上,求解异或方程组.方程组中的系数和常数为0或1,每个未知数的取值也为0或1
//存储答案在a.m[i][m]
// int xor_gauss(mat<int>& a, int n, int m) {
//     int r, c;//行,列
//     for (r = 1, c = 1;r <= n && c < m;r++, c++) {
//         int t = r;
//         for (int i = r + 1;i <= n;i++) {//找非0行,也可以改成找最大防误差
//             if (a.m[i][c]) {
//                 t = i;break;
//             }
//         }
//         if (t != r) swap(a.m[r], a.m[t]);
//         if (!a.m[r][c]) {//该列都为0
//             r--;continue;
//         }
//         for (int i = r + 1;i <= n;i++) {
//             if (a.m[i][c]) for (int j = c;j <= m;j++) a.m[i][j] ^= a.m[r][j];
//         }
//     }
//     r--;
//     if (r < m - 1) {
//         for (int i = r + 1;i < m;i++) {
//             if (a.m[i][c]) return -1;//无解. 存在0=非0
//         }
//         return (m - 1) - r;//无穷多解,返回自由元数量.(总共m-1个变量)
//     }
//     for (int i = m - 1;i >= 1;i--) {
//         for (int j = i + 1;j <= m - 1;j++) a.m[i][m] ^= a.m[i][j] * a.m[j][m];//这里可以改用&符号
//     }
//     return 0;//唯一解
// }


//消成对角矩阵
//精度更高,代码更短.(更主要用来求解线性方程组)但是无法判断是无解还是无穷解
//n行n+1列矩阵(n+1中第n+1列是增广矩阵,即目的是求解线性方程组)
bool gauss_jordan(mat<ld>& a, int n) {
    for (int i = 1;i <= n;i++) {//枚举行列
        int r = i;
        for (int k = i;k <= n;k++) {//找非0行
            if (abs(a.m[k][i]) > eps) {
                r = k;break;
            }
        }
        if (r != i) swap(a.m[r], a.m[i]);//交换两行
        if (abs(a.m[i][i]) < eps) return 0;//无解或无穷多解
        for (int k = 1;k <= n;k++) {//对角化
            if (k == i) continue;
            if (abs(a.m[k][i]) < eps) continue;
            ld t = a.m[k][i] / a.m[i][i];
            for (int j = i;j <= n + 1;j++) {
                a.m[k][j] -= t * a.m[i][j];
            }
        }
    }
    for (int i = 1;i <= n;i++) {
        a.m[i][n + 1] /= a.m[i][i];//答案存储在a.m[i][m+1]
    }
    return 1;//唯一解
}

// //对模意义下的矩阵n*n求逆
// mat<int> res;
// bool mat_inv(mat<int>& a, int n) {
//     //对左半部分高斯约旦消元化成对角矩阵,最后化成单位矩阵.右边的单位矩阵变成该矩阵的逆矩阵
//     for (int i = 1;i <= n;i++) {//枚举行列
//         int r = i;
//         for (int k = i;k <= n;k++) {//找非0行
//             if (abs(a.m[k][i])) {
//                 r = k;break;
//             }
//         }
//         if (r != i) swap(a.m[r], a.m[i]);//交换两行
//         if (!abs(a.m[i][i])) return 0;//无解或无穷多解
//         int invii = inv(a.m[i][i]);
//         for (int k = 1;k <= n;k++) {//对角化
//             if (k == i) continue;
//             int t = a.m[k][i] * invii % MOD;//搭配modint
//             for (int j = i;j <= n * 2;j++) {
//                 a.m[k][j] -= t * a.m[i][j] % MOD;a.m[k][j] = norm(a.m[k][j]);
//             }
//         }
//         for (int j = 1;j <= 2 * n;j++) a.m[i][j] = a.m[i][j] * invii % MOD;
//     }
//     return 1;//唯一解
// }

```

### 矩阵（新）

``` cpp
const int maxn = 210;
const int MOD = 1e9 + 7;
struct matrix {
    int n;
    int data[maxn][maxn];
    matrix() = default;
    matrix(int n) :n(n) {}
    int* operator[] (int idx) { return data[idx]; }
    void dbg() {
        cout << "-----matrix:begin-----\n";
        for (int i = 1;i <= n;i++) {
            for (int j = 1;j <= n;j++) {
                cout << data[i][j] << ' ';
            }
            cout << endl;
        }
        cout << "-----matrix:end-----\n";
    }
};
matrix operator*(matrix& A, matrix& B) {
    int n = A.n;
    matrix C(n);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            for (int k = 1; k <= n; k++) {
                C[i][j] += A[i][k] * B[k][j] % MOD;
                if (C[i][j] >= MOD) C[i][j] -= MOD;
            }

    return C;
}
matrix operator+(matrix& A, matrix& B) {
    int n = A.n;
    matrix C(n);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            for (int k = 1; k <= n; k++) {
                C[i][j] = A[i][k] * B[k][j];
                if (C[i][j] >= MOD) C[i][j] -= MOD;
            }

    return C;
}
matrix qp(matrix A, int m) {
    int n = A.n;
    matrix ans(n);
    for (int i = 1; i <= n; i++) ans[i][i] = 1;
    while (m) {
        if (m & 1) ans = ans * A;
        m >>= 1;A = A * A;
    }
    return ans;
}
matrix pow_sum(matrix A, int m) {//qp(A,1)+qp(A,2)+...+qp(A,m) 
    int n = A.n;
    matrix ans(n), B = A;
    while (m) {
        if (m & 1) ans = ans * A + B;
        B = B * A + B;A = A * A;m >>= 1;
    }
    return ans;
}

//矩阵光速幂,处理同底数同模数的幂
namespace LP {
    ll getphi(ll x) {
        ll res = x;
        for (int i = 2; i * i <= x; i++) {
            if (x % i == 0) {
                res -= res / i;
                while (x % i == 0) x /= i;
            }
        }
        if (x > 1) res -= res / x;
        return res;
    }
    matrix base1[N], basesqrt[N];
    int Block_len;
    int Phi;
    ll maxn = 1e10;//模数的最大值
    void init(matrix x) {//初始化底数为x
        Phi = getphi(MOD);
        Block_len = sqrt(maxn) + 1;
        base1[0] = matrix(100);base1[0].I(); for (int i = 1;i <= Block_len;i++) base1[i] = base1[i - 1] * x;
        basesqrt[0] = matrix(100);basesqrt.I(); for (int i = 1;i <= Block_len;i++) basesqrt[i] = basesqrt[i - 1] * base1[Block_len];
    }
    matrix qp(unsigned long long x) {
        x %= Phi;
        return basesqrt[x / Block_len] * base1[x % Block_len];
    }
}
```

### 高斯消元

``` cpp
const int maxn = 110;
using ld = long double;
using matrix = ld[maxn][maxn];
using vect = array<ld, maxn>;
void matrix_clr(matrix a, int n, int m) {
    for (int i = 0;i <= n;i++) {
        for (int j = 0;j <= m;j++) {
            a[i][j] = 0;
        }
    }
}
void gauss_elimination(matrix A, int n) { //A的大小n*(n+1)，如果方程有唯一解则算出,否则出错
    for (int i = 0; i < n; ++i) {
        int r = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(A[j][i]) > fabs(A[r][i]))
                r = j;
        if (r != i) for (int j = 0; j <= n; ++j)
            swap(A[r][j], A[i][j]);
        for (int k = i + 1; k < n; ++k)
            for (int j = n; j >= i; --j)
                A[k][j] -= A[k][i] / A[i][i] * A[i][j];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            A[i][n] -= A[j][n] * A[i][j];
        A[i][n] /= A[i][i];
    }
}
//无解返回-1，有唯一解返回0，有无穷多解返回1。
//在有解的情况下通过ans返回任意一个解。
//矩阵A的大小为n * (m + 1)，表示有n个方程，m个变量。
const double eps = 1e-8;
int row[maxn], var[maxn];
int one_possible(matrix A, int n, int m, vect& ans) {
    memset(row, -1, sizeof(row));
    int r = 0;
    for (int c = 0; c < m && r < n; ++c) {
        int x = r;
        for (int i = x + 1; i < n; ++i)
            if (fabs(A[i][c]) > fabs(A[x][c]))
                x = i;
        if (x != r) for (int j = 0; j <= m; ++j)
            swap(A[x][j], A[r][j]);
        if (fabs(A[r][c]) < eps)
            continue;
        for (int k = r + 1; k < n; ++k)
            for (int j = m; j >= c; --j)
                A[k][j] -= A[k][c] / A[r][c] * A[r][j];
        row[c] = r++;
    }
    for (int i = r; i < n; ++i) if (fabs(A[i][m]) > eps)
        return -1;
    for (int c = m - 1; c >= 0; --c) {
        int x = row[c];
        if (x < 0)
            ans[c] = 0;
        else {
            for (int i = x - 1; i >= 0; --i)
                A[i][m] -= A[i][c] / A[x][c] * A[x][m];
            ans[c] = A[x][m] / A[x][c];
        }
    }
    return r < m;
}
```

### 线性规划单纯形法

```c++
const ld eps = 1e-10;
constexpr int MAXN = 405, MAXM = 405;
struct Simplex {
    inline static ld a[MAXN][MAXM], b[MAXN], c[MAXM];//矩阵a,约束条件b,
    inline static ld d[MAXN][MAXM], x[MAXM];
    inline static int ix[MAXN + MAXM];
    Simplex() {}
    ld run(int n, int m) {
        m++;
        for (int i = 0; i < m + 1; i++) d[n][i] = d[n + 1][i] = 0;
        for (int i = 0; i < n + m; i++) ix[i] = i;

        int r = n, s = m - 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m - 1; ++j) d[i][j] = -a[i][j];
            d[i][m - 1] = 1;
            d[i][m] = b[i];
            if (d[r][m] > d[i][m]) r = i;
        }
        for (int i = 0;i < m - 1;i++) d[n][i] = c[i];
        d[n + 1][m - 1] = -1;
        for (ld dd; ; ) {
            if (r < n) {
                swap(ix[s], ix[r + m]);
                d[r][s] = 1.0 / d[r][s];
                for (int j = 0;j <= m;j++) if (j != s) d[r][j] *= -d[r][s];
                for (int i = 0;i <= n + 1;i++) {
                    if (i != r) {
                        for (int j = 0;j <= m;j++) {
                            if (j != s) {
                                d[i][j] += d[r][j] * d[i][s];
                            }
                        }
                        d[i][s] *= d[r][s];
                    }
                }
            }
            r = s = -1;
            for (int j = 0;j < m;j++) {
                if (s < 0 || ix[s] > ix[j]) {
                    if (d[n + 1][j] > eps || (d[n + 1][j] > -eps && d[n][j] > eps)) s = j;
                }
            }
            if (s < 0) break;
            for (int i = 0; i < n; ++i) {
                if (d[i][s] < -eps) {
                    if (r < 0 || (dd = d[r][m] / d[r][s] - d[i][m] / d[i][s]) < -eps || (dd < eps && ix[r + m] > ix[i + m])) r = i;
                }
            }
            if (r < 0) return -1;//无解
        }
        if (d[n + 1][m] < -eps) return -1;
        ld ans = 0;
        for (int i = 0;i < m;i++) x[i] = 0;
        for (int i = m; i < n + m; i++) {
            if (ix[i] < m - 1) {
                ans += d[i - m][m] * c[ix[i]];
                x[ix[i]] = d[i - m][m];
            }
        }
        return ans;
    }
};
```



## 数据结构



### ST表

```c++
struct StaticTable {
    const int LOGN = 20;
    const int n;
    vector<vector<int>> f;

    StaticTable(const vector<int>& a) :n(a.size() - 1) {

        f.assign(LOGN + 1, vector<int>(n + 1));
        for (int i = 1; i <= n; i++) {
            f[0][i] = a[i];
        }
        for (int j = 1; j <= LOGN; j++) {
            for (int i = 1; i + (1 << j) - 1 <= n; i++) {
                f[j][i] = max(f[j - 1][i], f[j - 1][i + (1 << (j - 1))]);
            }
        }
    }

    int query(int l, int r) {
        int len = __lg(r - l + 1);
        return max(f[len][l], f[len][r - (1 << len) + 1]);
    }

};



struct StaticTable2D {
    int LOGN = 20;
    int LOGM = 20;
    const int n, m;
    vector<vector<vector<vector<int>>>> f;
    StaticTable2D(const vector<vector<int>>& a) : n(a.size() - 1), m(a[0].size() - 1) {
        LOGN = log2(n);
        LOGM = log2(m);
        f.assign(LOGN + 1, vector<vector<vector<int>>>(LOGM + 1, vector<vector<int>>(n + 1, vector<int>(m + 1))));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                f[0][0][i][j] = a[i][j];
            }
        }
        for (int i = 0; i <= LOGN; i++) {
            for (int j = 0; j <= LOGN; j++) {
                if (i == 0 && j == 0) continue;
                for (int x = 1; x + (1 << i) - 1 <= n; x++) {
                    for (int y = 1; y + (1 << j) - 1 <= m; y++) {
                        if (i == 0) {
                            f[i][j][x][y] = max(f[i][j - 1][x][y], f[i][j - 1][x][y + (1 << (j - 1))]);
                        }
                        else {
                            f[i][j][x][y] = max(f[i - 1][j][x][y], f[i - 1][j][x + (1 << (i - 1))][y]);
                        }
                    }
                }
            }
        }
    }

    int query(int x1, int y1, int x2, int y2) {
        int lenx = __lg(x2 - x1 + 1);
        int leny = __lg(y2 - y1 + 1);
        int ans = f[lenx][leny][x1][y1];
        ans = max(ans, f[lenx][leny][x2 - (1 << lenx) + 1][y1]);
        ans = max(ans, f[lenx][leny][x1][y2 - (1 << leny) + 1]);
        ans = max(ans, f[lenx][leny][x2 - (1 << lenx) + 1][y2 - (1 << leny) + 1]);
        return ans;
    }
};

```





### 并查集(DSU)

```c++
struct DSU {
    vector<int> p, siz;
    DSU(int n) :p(n), siz(n, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) {
        return p[x] == x ? x : p[x] = find(p[x]);
    }
    bool same(int x, int y) {
        return find(x) == find(y);
    }
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return 0;
        if (siz[x] < siz[y]) {
            swap(x, y);
        }
        siz[x] += siz[y];
        p[y] = x;
        return 1;
    }
    int size(int x) {
        return siz[find(x)];
    }
};
```

### 可撤销并查集

```c++
struct DSU
{
    int n = 0, tot = 0, fa[N], sz[N], s[N];
    void ins() { n++, fa[n] = n, sz[n] = 1; }          // 插入节点
    int F(int x) { return fa[x] == x ? x : F(fa[x]); } // 即find查找函数
    void U(int x, int y)
    { // 合并函数
        x = F(x), y = F(y);
        if (x == y)
            return;
        if (sz[x] < sz[y])
            swap(x, y);
        s[++tot] = y, fa[y] = x, sz[x] += sz[y];
    }
    void D()
    { // 删除栈顶边
        if (!tot)
            return;
        int y = s[tot--];
        sz[fa[y]] -= sz[y], fa[y] = y;
    }
    void back(int t = 0)
    {
        while (tot > t)
            D();
    } // 删除到只剩t条边
} d;
```



### 树状数组

```c++
struct BIT {
    int n;
    vector<int> tr;
    BIT(int n) : n(n), tr(n + 1) {}

    inline int lbt(int x) { return x & -x; }

    void add(int i, int val) {
        for (; i <= n; i += lbt(i)) {
            tr[i] += val;
        }
    }
    int sum(int i) {
        int res = 0;
        for (; i > 0; i -= lbt(i)) {
            res += tr[i];
        }
        return res;
    }
    int query(int l, int r) {
        if (l > r) return 0;
        return sum(r) - sum(l - 1);
    }
    int kth(int k) {
        int x = 0;
        int res = 0;
        for (int i = 1 << __lg(n); i > 0; i /= 2) {
            if (x + i <= n && k >= tr[x + i]) {
                x += i;
                k -= tr[x];
                res += tr[x];
            }
        }
        return (k ? -1 : res);
    }
};
```

### 树状数组查前缀和

```cpp
struct Fenwick{
    int n;
    vector<int> tr;

    Fenwick(int n) : n(n), tr(n + 1, 0){}

    void modify(int x, int c){
        for(int i = x; i <= n; i += lowbit(i)) tr[i] += c;
    }

    void modify(int l, int r, int c){
        modify(l, c);
        if (r + 1 <= n) modify(r + 1, -c);
    }

    int query(int x){
        int res = 0;
        for(int i = x; i; i -= lowbit(i)) res += tr[i];
        return res;
    }

    int query(int l, int r){
        return query(r) - query(l - 1);
    }

    int find_first(int sum){//前缀和第一个>=sum
        int ans = 0; int val = 0;
        for(int i = __lg(n); i >= 0; i--){
            if ((ans | (1 << i)) <= n && val + tr[ans | (1 << i)] < sum){
                ans |= 1 << i;
                val += tr[ans];
            }
        }
        return ans + 1;
    }

    int find_last(int sum){//数组里前缀和最后一个<=sum
        int ans = 0; int val = 0;
        for(int i = __lg(n); i >= 0; i--){
            if ((ans | (1 << i)) <= n && val + tr[ans | (1 << i)] <= sum){
                ans |= 1 << i;
                val += tr[ans];
            }
        }
        return ans;
    }

};
```





### 树状数组区间版

```
struct Fenwick_range{
    int n;
    vector<int> tr1,tr2;

    Fenwick_range(int n) : n(n), tr1(n + 1, 0),tr2(n+1,0) {}
    int lowbit(int x){
        return x & -x;
    }

    void modify(int x, int c,vector<int>&tr){
        for(int i = x; i <= n; i += lowbit(i)) tr[i] += c;
    }

    void modify(int l, int r, int c){
        modify(l, c, tr1),modify(l,c*(l-1),tr2);
        if(r+1<=n)
            modify(r + 1, -c, tr1), modify(r+1, -c * r, tr2);
    }
    int query(int x, vector<int> &tr)
    {
        int res = 0;
        for(int i = x; i; i -= lowbit(i)) res += tr[i];
        return res;
    }
    int query(int l, int r){
        return (query(r, tr1) * r - query(r, tr2)) - (query(l-1,tr1)*(l-1)-query(l-1,tr2));
    }
};
```



### 线段树

```c++
//封装·线段树
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Tag {
        ll add = 0;
    };
    struct Info {
        int sz;
        ll sum = 0;
    };
    struct node {
        Info info;
        Tag tag;
    };
    Info friend operator +(const Info& l, const Info& r) {
        return { l.sz + r.sz,l.sum + r.sum };
    }
    Info friend operator +(const Info& info, const Tag& tag) {
        return { info.sz,info.sum + tag.add * info.sz };
    }
    Tag friend operator+(const Tag& tag1, const Tag& tag2) {
        return { tag1.add + tag2.add };
    }
    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        tr[x].tag = { 0 };
        if (l == r) {
            tr[x].info = { 1,a[l] };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
    void settag(int x, Tag tag) {
        tr[x].info = tr[x].info + tag;
        tr[x].tag = tr[x].tag + tag;
    }
    void pushdown(int x) {//下传标记
        settag(ls, tr[x].tag);
        settag(rs, tr[x].tag);
        tr[x].tag.add = 0;
    }
    //l,r代表操作区间
    void update(int x, int l, int r, int ql, int qr, Tag tag) {
        if (l == ql && r == qr) {
            settag(x, tag);
            return;
        }
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) update(ls, l, mid, ql, qr, tag);
        else if (mid < ql) update(rs, mid + 1, r, ql, qr, tag);
        else {
            update(ls, l, mid, ql, mid, tag);
            update(rs, mid + 1, r, mid + 1, qr, tag);
        }
        pushup(x);
    }
    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }

};

     // int search(int x, int l, int r, int ql, int qr, int d) {//mx表示区间最大值,线段树二分,找到区间[ql,qr]第一个大于等于d的位置
    //     if (l == ql && r == qr) {
    //         if (tr[x].mx < d) return -1;
    //         if (l == r) return l;
    //         int mid = l + r >> 1;
    //         if (tr[ls].mx >= d) return search(ls, l, mid, ql, mid, d);
    //         else return search(rs, mid + 1, r, mid + 1, qr, d);
    //     }
    //     int mid = l + r >> 1;
    //     if (qr <= mid) return search(ls, l, mid, ql, qr, d);
    //     else if (mid < ql) return search(rs, mid + 1, r, ql, qr, d);
    //     else {
    //         int pos = search(ls, l, mid, ql, mid, d);
    //         if (pos == -1) return search(rs, mid + 1, r, mid + 1, qr, d);
    //         else return pos;
    //     }
    // }

```

### 线段树合并

```cpp
namespace tree_merge
{
    const int MAXN = 5e5 + 5;//值域范围
    int cnt = 0, sz = MAXN-5;//线段树值域大小
    struct info
    {
        ull n1 = 0, n2 = 0, n3 = 0;
        info() {}
        info(ull n1, ull n2, ull n3) : n1(n1), n2(n2), n3(n3) {}
        void modify(info a)
        {
            n1 += a.n1;
            n2 += a.n2;
            n3 += a.n3;
        }
        void clear()
        {
            n1 = n2 = n3 = 0;
        }
    };
    const info empty = info();
    int stk[MAXN << 3], top = 0;
    struct node
    {
        int l, r;
        info nu = empty;
        void clear()
        {
            nu.clear();
            l = r = 0;
        }
    } tr[MAXN << 3];
    struct trees
    {
        int root;
    } tree[MAXN<<3];
    void Tpush_up(int &p)
    { // 合并答案
        tr[p].nu.clear();
        if (tr[p].l)
            tr[p].nu.modify(tr[tr[p].l].nu);
        if (tr[p].r)
            tr[p].nu.modify(tr[tr[p].r].nu);
    }
    int Tnew()
    { // 动态开点
        if (top)
            return stk[top--];

        return ++cnt;
    }
    void Tdelete(int &p)
    { // 内存回收
        tr[p].clear();
        stk[++top] = p;
    }
    void Tmodify(int &p, int l, int r, int pos, info &x)
    {
        if (!p)
            p = Tnew();
        if (l == r)
        {
            tr[p].nu.modify(x);
            // cout << tr[p].nu.n1 << endl;
            return;
        }
        int mid = l + r >> 1;
        if (pos <= mid)
            Tmodify(tr[p].l, l, mid, pos, x);
        else
            Tmodify(tr[p].r, mid + 1, r, pos, x);
        Tpush_up(p);
    }
    int Tmerge(int p1, int p2, int l, int r)
    {
        if (!p1 || !p2)
            return p1 + p2;
        int p = Tnew();
        if (l == r)
        {
            tr[p].nu = tr[p1].nu, tr[p].nu.modify(tr[p2].nu);
            Tdelete(p1);
            Tdelete(p2);
            return p;
        }
        int mid = l + r >> 1;
        tr[p].l = Tmerge(tr[p1].l, tr[p2].l, l, mid);
        tr[p].r = Tmerge(tr[p1].r, tr[p2].r, mid + 1, r);
        Tpush_up(p);
        Tdelete(p1);
        Tdelete(p2);
        return p;
    }
    info Tquery(int &p, int l, int r, int ql, int qr)
    {
        if (!p)
            return empty;
        if (l >= ql && r <= qr)
        {
            return tr[p].nu;
        }
        int mid = l + r >> 1;
        info ans;
        if (ql <= mid)
            ans.modify(Tquery(tr[p].l, l, mid, ql, qr));
        if (qr > mid)
            ans.modify(Tquery(tr[p].r, mid + 1, r, ql, qr));
        return ans;
    }
    // tree_merge(int size)
    // { // 构造函数
    //     sz = size;
    // }
    void merge(int a, int b)
    { // 把第b颗树合并到第a颗里面
        tree[a].root = Tmerge(tree[a].root, tree[b].root, 1, sz);
    }
    info query(int a, int l, int r)
    { // 询问第a颗树l,r区间内的信息
        if (l > r)
            return empty;
        return Tquery(tree[a].root, 1, sz, l, r);
    }
    void modify(int a, int pos, info x)
    { // 在点pos上加上x
        // cout << x.n1 << endl;
        Tmodify(tree[a].root, 1, sz, pos, x);
    }
    void init(int n){//初始化，指定线段树值域范围
        sz=n;
        for(int i=1;i<=cnt;i++)tr[i].clear();
        cnt=0;top=0;
    }
};
```

### 线段树树上二分

```cpp
namespace ST
{
    const int MAXN = 5e5 + 5;
    int sz = MAXN - 5;
    struct info
    {
        int sum = 0;
        info() {}
        info(int sum) : sum(sum) {}
        void modify(info a)
        {
            sum += a.sum;
        }
        void clear()
        {
            sum = 0;
        }
        bool equal(info &a)
        { // 等于运算
            return sum == a.sum;
        }
        bool cmp(info &a)
        { // 大于运算
            return sum > a.sum;
        }
    };
    void Tmax(info &a, info &b)
    {
        if (b.cmp(a))
            a = b;
    }
    const info empty = info();
    struct node
    {
        info nu, ma;
    } tr[MAXN << 2];
    void push_up(int p)
    {
        tr[p].nu.clear();
        tr[p].ma.clear();
        tr[p].nu.modify(tr[p << 1].nu);
        tr[p].nu.modify(tr[p << 1 | 1].nu);
        Tmax(tr[p].ma, tr[p << 1].ma);
        Tmax(tr[p].ma, tr[p << 1 | 1].ma);
    }
    void build(int p, int l, int r, vector<info> &v)
    {
        if (l == r)
        {
            tr[p].nu = v[l];
            tr[p].ma = v[l];
            return;
        }
        int mid = l + r >> 1;
        build(p << 1, l, mid, v);
        build(p << 1 | 1, mid + 1, r, v);
        push_up(p);
    }

    void modify(int p, int l, int r, int pos, info x)
    {
        if (l == r)
        {
            tr[p].nu.modify(x);
            tr[p].ma.modify(x);
            return;
        }
        // push_down(p);
        int mid = l + r >> 1;
        if (pos <= mid)
            modify(p << 1, l, mid, pos, x);
        else
            modify(p << 1 | 1, mid + 1, r, pos, x);
        push_up(p);
    }
    int find_first(int p, int l, int r, int ql, int qr, info x)
    { // 某个区间左边开始第一个比x大的
        if (r < ql || l > qr || !tr[p].ma.cmp(x))
            return -1;
        if (l == r)
        {
            if (tr[p].ma.cmp(x))
                return l;
            return -1;
        }
        // push_down(p);
        int mid = l + r >> 1;
        int pos = find_first(p << 1, l, mid, ql, qr, x);
        if (pos == -1)
            pos = find_first(p << 1 | 1, mid + 1, r, ql, qr, x);
        return pos;
    }
    int find_last(int p, int l, int r, int ql, int qr, info x)
    { // 某个区间右边边开始第一个比x大的
        if (r < ql || l > qr || !tr[p].ma.cmp(x))
            return -1;
        if (l == r)
        {
            if (tr[p].ma.cmp(x))
                return l;
            return -1;
        }
        // push_down(p);
        int mid = l + r >> 1;
        int pos = find_last(p << 1 | 1, mid + 1, r, ql, qr, x);
        if (pos == -1)
            pos = find_last(p << 1, l, mid, ql, qr, x);
        return pos;
    }
    info query(int p, int l, int r, int ql, int qr)
    {
        if (l >= ql && r <= qr)
        {
            return tr[p].nu;
        }
        int mid = l + r >> 1;
        info ans;
        if (ql <= mid)
            ans.modify(query(p << 1, l, mid, ql, qr));
        if (qr > mid)
            ans.modify(query(p << 1 | 1, mid + 1, r, ql, qr));
        return ans;
    }
}
void solve()
{
    int n, q;
    cin >> n >> q;
    vector<ST::info> v(n + 2);
    for (int i = 1; i <= n; i++)
        cin >> v[i].sum;
    ST::build(1, 1, n, v);
    while (q--)
    {
        int op;
        cin >> op;
        if (op == 1)
        {
            int x, k;
            cin >> x >> k;
            ST::modify(1, 1, n, x, ST::info(k));
        }
        else
        {
            int l, r;
            cin >> l >> r;
            ST::info nu = ST::query(1, 1, n, l, r);
            cout << nu.sum << endl;
        }
    }
}
```



### 线段树维护摩尔投票

由于绝对众数可能不存在，还需要检验一下绝对众数是否存在，可以对每个权值预处理个下标数组然后二分。

```c++
//封装·线段树
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
 
    struct Info {
        int sz;
        int m = 0;
        ll sum = 0;
    };
    struct node {
        Info info;
    };
    Info friend operator +(const Info& l, const Info& r) {
        if (l.m == r.m) return { l.sz + r.sz,l.m,l.sum + r.sum };
        else if (l.sum < r.sum) return { l.sz + r.sz,r.m,r.sum - l.sum };
        else return { l.sz + r.sz,l.m,l.sum - r.sum };
    }
 
    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        if (l == r) {
            tr[x].info = { 1,a[l],1 };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
 
    void update(int x, int l, int r, int q, int k) {
        if (l == q && r == q) {
            tr[x].info.m = k;
            return;
        }
        int mid = l + r >> 1;
        if (q <= mid) update(ls, l, mid, q, k);
        else if (mid < q) update(rs, mid + 1, r, q, k);
        pushup(x);
    }
 
    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }
 
};
//检验绝对众数是否合法
auto itl = lower_bound(pos[m].begin(), pos[m].end(), right);
auto itr = upper_bound(pos[m].begin(), pos[m].end(), right);
if (itr - itl > (right - left + 1) / 2) return m;
else return -1;
```

### 带权扫描线线段树

``` cpp
struct segtree
{
    struct node
    {
        int mi, num, add;
    } tr[MAXN << 2];
    void build(int p, int l, int r)
    {
        if (l == r)
        {
            tr[p].mi = INF;
            return;
        }
        int mid = l + r >> 1;
        build(p << 1, l, mid);
        build(p << 1 | 1, mid + 1, r);
        push_up(p);
    }
    void push_up(int p)
    {
        if (tr[p << 1].mi == tr[p << 1 | 1].mi)
        {
            tr[p].mi = tr[p << 1].mi;
            tr[p].num = (tr[p << 1].num + tr[p << 1 | 1].num) % mod;
        }
        else if (tr[p << 1].mi < tr[p << 1 | 1].mi)
        {
            tr[p].mi = tr[p << 1].mi;
            tr[p].num = tr[p << 1].num;
        }
        else
        {
            tr[p].mi = tr[p << 1 | 1].mi;
            tr[p].num = tr[p << 1 | 1].num;
        }
    }
    void push_down(int p)
    {
        if (!tr[p].add)
            return;
        tr[p << 1].mi += tr[p].add;
        tr[p << 1 | 1].mi += tr[p].add;
        tr[p << 1].add += tr[p].add;
        tr[p << 1 | 1].add += tr[p].add;
        tr[p].add = 0;
    }
    void modify(int p, int l, int r, int pos, int nu)
    {
        if (l == r)
        {
            tr[p].mi = 0;
            tr[p].num = nu;
            return;
        }
        int mid = l + r >> 1;
        push_down(p);
        if (pos <= mid)
            modify(p << 1, l, mid, pos, nu);
        else
            modify(p << 1 | 1, mid + 1, r, pos, nu);
        push_up(p);
    }
    void add(int p, int l, int r, int ql, int qr, int nu)
    {
        if (l >= ql && r <= qr)
        {
            tr[p].mi += nu;
            tr[p].add += nu;
            return;
        }
        int mid = l + r >> 1;
        push_down(p);
        if (ql <= mid)
            add(p << 1, l, mid, ql, qr, nu);
        if (qr > mid)
            add(p << 1 | 1, mid + 1, r, ql, qr, nu);
        push_up(p);
    }
	int query(int p,int l,int r,int ql,int qr){//查询非空段数量
		if (l >= ql && r <= qr){
			if(tr[p].mi==0)return tr[p].num;
			else return 0;
		}
		  int mid = l + r >> 1;
        push_down(p);
        if (ql <= mid)
           	query(p << 1, l, mid, ql, qr);
        if (qr > mid)
            query(p << 1 | 1, mid + 1, r, ql, qr);
        push_up(p);
	}
} tree;
```



### 线段树维护区间最大子段和

```c++
 //区间最大子段和+单点修改
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Tag {
        ll add = 0;
    };
    struct Info {
        int sz;
        ll sum;
        ll premax;
        ll sufmax;
        ll mx;
    };
    struct node {
        Info info;
        Tag tag;
    };
    Info friend operator +(const Info& l, const Info& r) {
        return { l.sz + r.sz,l.sum + r.sum,max(l.sum + r.premax,l.premax),max(l.sufmax + r.sum,r.sufmax),max({l.mx,r.mx,l.sufmax + r.premax}) };
    }
    Info friend operator +(const Info& info, const Tag& tag) {
        return { info.sz,tag.add,tag.add,tag.add,tag.add };
    }
    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        tr[x].tag = { 0 };
        if (l == r) {
            tr[x].info = { 1,a[l],a[l],a[l],a[l] };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
    void settag(int x, Tag tag) {
        tr[x].info = tr[x].info + tag;
    }
    //l,r代表操作区间
    void update(int x, int l, int r, int q, Tag tag) {
        if (l == r) {
            settag(x, tag);
            return;
        }
        int mid = l + r >> 1;
        if (q <= mid) update(ls, l, mid, q, tag);
        else  update(rs, mid + 1, r, q, tag);
        pushup(x);
    }
    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }
 
};



//其二
 struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Info {
        int sz;
        ll sum;//区间和
        ll premax, sufmax, mx;//前缀最大，后缀最大，区间最大
        ll premin, sufmin, mn;//前缀最小，后缀最小，区间最小
        ll bg_res, ed_res;//前缀答案，后缀答案
        ll all_res;//区间整体答案
        ll res;//答案
    };
    struct node {
        Info info;
    };
    Info friend operator +(const Info& l, const Info& r) {
        return { l.sz + r.sz,l.sum + r.sum,
        max(l.sum + r.premax,l.premax),max(l.sufmax + r.sum,r.sufmax),max({l.mx,r.mx,l.sufmax + r.premax}),
        min(l.sum + r.premin,l.premin),min(l.sufmin + r.sum,r.sufmin),min({l.mn,r.mn,l.sufmin + r.premin}),
        max({l.bg_res,l.all_res - r.premin,l.sum - r.premin,l.sum + r.bg_res }), max({r.ed_res,l.sufmax + r.all_res,l.sufmax - r.sum,l.ed_res - r.sum}),
        max({l.sum - r.sum,l.all_res - r.sum,l.sum + r.all_res}),
        max({l.sufmax - r.premin,l.res,r.res,l.ed_res - r.premin,l.sufmax + r.bg_res})
        };
    }
    Info friend operator +(const Info& info, const int x) {
        return { info.sz,x,x,x,x,x,x,x,-inf,-inf,-inf,-inf };
    }
    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        if (l == r) {
            tr[x].info = { 1,a[l],a[l],a[l],a[l],a[l],a[l],a[l],-inf,-inf,-inf,-inf };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
    void settag(int x, int tag) {
        tr[x].info = tr[x].info + tag;
    }
    //l,r代表操作区间
    void update(int x, int l, int r, int q, int tag) {
        if (l == r) {
            settag(x, tag);
            return;
        }
        int mid = l + r >> 1;
        if (q <= mid) update(ls, l, mid, q, tag);
        else  update(rs, mid + 1, r, q, tag);
        pushup(x);
    }
    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }
 
};
 
void Solve(int TIME) {
 
    int n, q;cin >> n >> q;
    vi a(n + 1);for (int i = 1;i <= n;i++) cin >> a[i];
    Segtree tr(a, n);
    while (q--) {
        int op;cin >> op;
        if (op == 1) {
            int x, y;cin >> x >> y;
            tr.update(1, 1, n, x, y);
        }
        else {
            int l, r;cin >> l >> r;
            cout << tr.query(1, 1, n, l, r).res << endl;
        }
    }
 
 
}
```

### 线段树维护线段并集之和（扫描线线段树）



``` cpp
struct SMT{ /// just a SegMent 
    struct Node{
        int val;
        int sum;
    
        Node(int _val=0, int _sum=0){
            val=_val;
            sum=_sum;
        }
    };
    int n;
    vector<Node> tree;
    SMT(int _n=0): n(_n) {
        tree.assign(n*4+5, Node());
    }
    void update(int l, int r, int id, int u, int v, int val){
        if (l>v || r<u) return;
        if (l>=u && r<=v){
            int _id= id<<1;
            tree[id].val+=val;
            if (tree[id].val==0){
                if (l!=r) tree[id].sum = tree[_id].sum + tree[_id|1].sum;
                else tree[id].sum=0;
            }
            else tree[id].sum = r-l+1;
            return;
        }
        int mid = (l+r)>>1;
        int _id = id<<1;
        update(l, mid, _id, u, v, val);
        update(mid+1, r, _id|1, u, v, val);
        if (tree[id].val==0) tree[id].sum = tree[_id].sum+tree[_id|1].sum;
    }
    int get(){
        return tree[1].sum;
    }
    void update(int u, int v, int val){
        update(1, n, 1, u, v, val);
    }
};
```



### 主席树1

```c++
const int MAXN = 2e5 + 5;

int head[MAXN];
struct segment_tree
{
    int cnt = 0, n; // 点数，线段树大小
    vector<int> nu; // 值域数组，第i个数表示在线段树中去离散后为i的数的真实值
    struct node
    {
        int l, r, val, sum;
    } tr[MAXN << 5];
    int copy(node &p)
    {
        tr[++cnt] = p;
        return cnt;
    }
    void init(vector<int> &t)//注意，先init再build
    {
        nu = t;
        n = nu.size();
        cnt = 0;
    }
    int build(int l, int r)
    {
        int p = ++cnt;
        tr[p].val = 0;
        tr[p].sum = 0;
        if (l < r)
        {
            int mid = l + r >> 1;
            tr[p].l = build(l, mid);
            tr[p].r = build(mid + 1, r);
        }
        return p;
    }
    int add(int p, int l, int r, int addpos, int x) // 操作并返回新线段树的head
    {
        int t = copy(tr[p]), mid = (l + r >> 1);
        tr[t].sum += nu[addpos] * x;
        tr[t].val += x;
        if (l < r)
        {
            if (addpos <= mid)
                tr[t].l = add(tr[p].l, l, mid, addpos, x);
            else
                tr[t].r = add(tr[p].r, mid + 1, r, addpos, x);
        }
        return t;
    }
    int query_more(int ql, int qr, int l, int r, int k) // ql到qr区间大于等于k的数
    {
        if (r < k)
            return 0;
        if (l >= k)
        {
            return tr[qr].val - tr[ql].val;
        }
        int mid = l + r >> 1, sum = 0;
        if (mid >= k)
            sum += query_more(tr[ql].l, tr[qr].l, l, mid, k);
        sum += query_more(tr[ql].r, tr[qr].r, mid + 1, r, k);
        return sum;
    }
    int query_less(int ql, int qr, int l, int r, int k) // ql到qr区间小于等于k的数
    {
        if (l > k)
            return 0;
        if (r <= k)
        {
            return tr[qr].val - tr[ql].val;
        }
        int mid = l + r >> 1, sum = 0;
        sum += query_less(tr[ql].l, tr[qr].l, l, mid, k);
        if (mid + 1 <= k)
            sum += query_less(tr[ql].r, tr[qr].r, mid + 1, r, k);
        return sum;
    }

    int query_kth(int ql, int qr, int l, int r, int k) // 询问ql到qr之间的数的第k小
    {
        if (l >= r)
            return l; // 返回下标
        int value = tr[tr[qr].l].val - tr[tr[ql].l].val, mid = l + r >> 1;
        if (value >= k)
            return query_kth(tr[ql].l, tr[qr].l, l, mid, k);
        else
            return query_kth(tr[ql].r, tr[qr].r, mid + 1, r, k - value);
    }
    int query_ksum(int ql, int qr, int l, int r, int k) // 询问ql+1到qr之间的数的前k小数之和
    {
        if (l >= r)
            return (k)*nu[l]; // 直接返回值

        int value = tr[tr[qr].l].val - tr[tr[ql].l].val, mid = l + r >> 1;
        // cout << value << endl;
        if (value >= k)
            return query_ksum(tr[ql].l, tr[qr].l, l, mid, k);
        else
            return query_ksum(tr[ql].r, tr[qr].r, mid + 1, r, k - value) + (tr[tr[qr].l].sum - tr[tr[ql].l].sum);
    }
    int query_nomorethan(int ql, int qr, int l, int r, int k)
    { // ql+1到qr之间以小的优先加起来不超过k的数的数量
        if (l >= r)
        {
            return (k <= 0ll ? 0ll : min(k / nu[l],tr[qr].val-tr[ql].val));
        }
        int mid = l + r >> 1, value = tr[tr[qr].l].sum - tr[tr[ql].l].sum;
        if (value >= k)
            return query_nomorethan(tr[ql].l, tr[qr].l, l, mid, k);
        else
            return tr[tr[qr].l].val - tr[tr[ql].l].val + query_nomorethan(tr[ql].r, tr[qr].r, mid + 1, r, k - value);
    }
    int query(int l, int r, int k)
    {
        return query_ksum(head[l - 1], head[r], 1, n, k);
    }
} SGT;
```



### 主席树2

```c++
//可持久化权值线段树(主席树)
struct PST {
#define ls(x) (tr[x].son[0])
#define rs(x) (tr[x].son[1])
    struct node {
        int son[2] = { 0,0 };
        ll sum_val = 0;
        int sum_cnt = 0;
    };
    static node tr[N << 5];//n+Qlogn
    vector<int> v;//去离散化的原数组
    inline static int root[N];//n+Q,第i个版本的根节点编号
    int n, idx;//总值域大小,当前节点数
    PST(int n) :n(n), idx(0) {
        v.resize(n + 1);
        build(root[0], 1, n);
    }
    ~PST() {
        for (int i = 1;i <= idx;i++) tr[i].sum_cnt = tr[i].sum_val = 0;
        idx = 0;
    }
    void build(int& x, int l, int r) {
        x = ++idx;
        if (l == r) return;
        int mid = l + r >> 1;
        build(ls(x), l, mid);
        build(rs(x), mid + 1, r);
    }
    int copy(int x) {
        tr[++idx] = tr[x];
        return idx;
    }
    void pushup(int x) {
        tr[x].sum_val = tr[ls(x)].sum_val + tr[rs(x)].sum_val;
        tr[x].sum_cnt = tr[ls(x)].sum_cnt + tr[rs(x)].sum_cnt;
    }
    //在x版本的基础上给离散化后第k个数字的数量加cnt
    void insert(int& now, int pre, int l, int r, int k, int cnt) {
        now = copy(pre);//如果修改旧节点,去掉
        if (l == r) {
            tr[now].sum_cnt += cnt;
            tr[now].sum_val += v[k] * cnt;
            return;
        }
        int mid = l + r >> 1;
        if (k <= mid) insert(ls(now), ls(pre), l, mid, k, cnt);
        else insert(rs(now), rs(pre), mid + 1, r, k, cnt);
        pushup(now);
    }
    //版本x中值域[ql,qr]的数的数量
    int query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return  tr[x].sum_cnt;
        int mid = l + r >> 1;
        if (qr <= mid) return query(ls(x), l, mid, ql, qr);
        else if (mid < ql) return query(rs(x), mid + 1, r, ql, qr);
        else return query(ls(x), l, mid, ql, mid) + query(rs(x), mid + 1, r, mid + 1, qr);
    }
    //数组中下标在[ql+1,qr]内的第k小值,记得加root
    int kth_min(int l, int r, int ql, int qr, int k) {
        if (l == r) return v[l];
        int mid = l + r >> 1;
        int s = tr[ls(qr)].sum_cnt - tr[ls(ql)].sum_cnt;
        if (k <= s) return kth_min(l, mid, ls(ql), ls(qr), k);
        else return kth_min(mid + 1, r, rs(ql), rs(qr), k - s);
    }
    int kth_max(int l, int r, int ql, int qr, int k) {
        if (l == r) return v[l];
        int mid = l + r >> 1;
        int s = tr[rs(qr)].sum_cnt - tr[rs(ql)].sum_cnt;
        if (k <= s) return kth_max(mid + 1, r, rs(ql), rs(qr), k);
        else return kth_max(l, mid, ls(ql), ls(qr), k - s);
    }
    //数组下标[ql+1,qr]之间前k小数之和
    ll kth_min_sum(int l, int r, int ql, int qr, int k) {
        if (l == r) return 1ll * k * v[l];
        int mid = l + r >> 1;
        int s = tr[ls(qr)].sum_cnt - tr[ls(ql)].sum_cnt;
        if (k <= s) return kth_min_sum(l, mid, ls(ql), ls(qr), k);
        else return kth_min_sum(mid + 1, r, rs(ql), rs(qr), k - s) + (tr[ls(qr)].sum_val - tr[ls(ql)].sum_val);
    }
    ll kth_max_sum(int l, int r, int ql, int qr, int k) {
        if (l == r) return 1ll * k * v[l];
        int mid = l + r >> 1;
        int s = tr[rs(qr)].sum_cnt - tr[rs(ql)].sum_cnt;
        if (k <= s) return kth_max_sum(mid + 1, r, rs(ql), rs(qr), k);
        else return kth_max_sum(l, mid, ls(ql), ls(qr), k - s) + (tr[rs(qr)].sum_val - tr[rs(ql)].sum_val);
    }

    int less_equall_x_cnt(int l, int r, int ql, int qr, int x) {
        if (l == r) return tr[qr].sum_cnt - tr[ql].sum_cnt;
        int ans = 0;
        int mid = l + r >> 1;
        if (x <= mid) ans += less_equall_x_cnt(l, mid, ls(ql), ls(qr), x);
        else {
            ans += tr[ls(qr)].sum_cnt - tr[ls(ql)].sum_cnt;
            ans += less_equall_x_cnt(mid + 1, r, rs(ql), rs(qr), x);
        }
        return ans;
    }
    ll less_equall_x_sum(int l, int r, int ql, int qr, int x) {
        if (l == r) return tr[qr].sum_val - tr[ql].sum_val;
        int ans = 0;
        int mid = l + r >> 1;
        if (x <= mid) ans += less_equall_x_sum(l, mid, ls(ql), ls(qr), x);
        else {
            ans += tr[ls(qr)].sum_val - tr[ls(ql)].sum_val;
            ans += less_equall_x_sum(mid + 1, r, rs(ql), rs(qr), x);
        }
        return ans;
    }

};
PST::node PST::tr[N << 5];
/*
PST pst(tot);pst.v = v;

for (int i = 1;i <= n;i++) {
    pst.insert(pst.root[i], pst.root[i - 1], 1, tot, val, 1);
}

pst.query(1, tot, pst.root[l - 1], pst.root[r]);//[l,r]
*/
```



### 线段树维护矩阵

```c++
//线段树维护矩阵
struct Mat {
    int m[3][3];
    Mat() {
        memset(m, 0, sizeof m);
    }
    Mat(int epsilon) {
        memset(m, 0, sizeof m);
        for (int i = 1;i <= 2;i++) m[i][i] = 1;
    }
    int& operator()(int i, int j) { return m[i][j]; }
    int operator()(int i, int j)const { return m[i][j]; }
    int is_I() {
        for (int i = 1;i <= 2;i++) {
            for (int j = 1;j <= 2;j++) {
                if (i == j && m[i][j] != 1) return 0;
                if (i != j && m[i][j] != 0) return 0;
            }
        }
        return 1;
    }
};
Mat I(1);
//矩阵乘法
Mat operator *(const Mat& x, const Mat& y) {
    Mat t;
    for (int k = 1;k <= 2;k++) {
        for (int i = 1;i <= 2;i++) {
            for (int j = 1;j <= 2;j++) {
                t.m[i][j] = t.m[i][j] + x.m[i][k] * y.m[k][j] % MOD;
                if (t.m[i][j] >= MOD) t.m[i][j] %= MOD;
            }
        }
    }
    return t;
}
//矩阵快速幂
Mat qp(Mat a, int k) {
    Mat res(1);
    while (k) {
        if (k & 1) res = res * a;
        a = a * a;
        k >>= 1;
    }
    return res;
}

struct Vec {
    int v[3];
    Vec() { memset(v, 0, sizeof v); }
    int operator()(int k)const { return v[k]; }
    int& operator()(int k) { return v[k]; }
};
//向量加法
Vec operator +(const Vec& x, const Vec& y) {
    Vec res;
    for (int i = 1;i <= 2;i++) {
        res(i) = (x(i) + y(i)) % MOD;
    }
    return res;
}
//向量乘矩阵
Vec operator*(const Vec& v, const Mat& m) {
    Vec res;
    for (int i = 1;i <= 2;i++) {
        for (int j = 1;j <= 2;j++) {
            res(j) = res(j) + v(i) * m(i, j) % MOD;
            if (res(j) >= MOD) res(j) %= MOD;
        }
    }
    return res;
}
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Tag {
        Mat add;
    };
    struct Info {
        int sz;
        Mat M;
    };
    struct node {
        Info info;
        Tag tag;
    };
    Info friend operator +(const Info& l, const Info& r) {
        return { l.sz + r.sz,l.M * r.M };
    }
    Info friend operator +(const Info& info, const Tag& tag) {
        return { info.sz,info.M * tag.add };
    }
    Tag friend operator+(const Tag& tag1, const Tag& tag2) {
        return { tag1.add * tag2.add };
    }
    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        tr[x].info.sz = r - l + 1;
        tr[x].tag = { I };
        if (l == r) {
            Mat m;
            tr[x].info = { 1,m };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }
    void settag(int x, Tag tag) {
        tr[x].info = tr[x].info + tag;
        tr[x].tag = tr[x].tag + tag;
    }
    void pushdown(int x) {//下传标记
        if (tr[x].tag.add.is_I() == 0) {
            settag(ls, tr[x].tag);
            settag(rs, tr[x].tag);
            tr[x].tag.add = I;
        }
    }
    //l,r代表操作区间
    void update(int x, int l, int r, int ql, int qr, Tag tag) {
        if (l == ql && r == qr) {
            settag(x, tag);
            return;
        }
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) update(ls, l, mid, ql, qr, tag);
        else if (mid < ql) update(rs, mid + 1, r, ql, qr, tag);
        else {
            update(ls, l, mid, ql, mid, tag);
            update(rs, mid + 1, r, mid + 1, qr, tag);
        }
        pushup(x);
    }
    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }

};
```



### 树状数组套主席树

```c++
//BIT套PST
struct PST {
#define ls(x) (tr[x].son[0])
#define rs(x) (tr[x].son[1])
    struct node {
        int son[2] = { 0,0 };
        int sum_cnt = 0;
        int sum_val = 0;
    };
    inline int lbt(int x) { return x & -x; }
    node tr[N * 256];//n+Qlogn
    vector<int> v;//去离散化的原数组
    inline static int root[N * 2];//第i个版本的根节点编号
    inline static int t1[N], t2[N];int n1 = 0, n2 = 0;
    int n, tot, idx;//数组范围,值域,当前节点数
    PST(int n, int tot) :idx(0), n(n), tot(tot) {
        v.resize(tot + 1);
        build(root[0], 1, tot);
    }
    void build(int& x, int l, int r) {
        x = ++idx;
        if (l == r) return;
        int mid = l + r >> 1;
        build(ls(x), l, mid);
        build(rs(x), mid + 1, r);
    }
    int copy(int x) {
        tr[++idx] = tr[x];
        return idx;
    }
    void pushup(int x) {
        tr[x].sum_cnt = tr[ls(x)].sum_cnt + tr[rs(x)].sum_cnt;
        tr[x].sum_val = tr[ls(x)].sum_val + tr[rs(x)].sum_val;
    }
    //在x版本的基础上给离散化后第k个数字的数量加cnt
    void insert(int& now, int pre, int l, int r, int k, int cnt) {
        if (!now) now = copy(pre);//这里只涉及修改自身，所以可以!now节省内存
        if (l == r) {
            tr[now].sum_cnt += cnt;
            tr[now].sum_val += v[k] * cnt;
            return;
        }
        int mid = l + r >> 1;
        if (k <= mid) insert(ls(now), ls(pre), l, mid, k, cnt);
        else insert(rs(now), rs(pre), mid + 1, r, k, cnt);
        pushup(now);
    }

    int kth_min(int l, int r, int k) {//第k小值
        if (l == r) return v[l];
        int mid = l + r >> 1;
        int s = 0;
        for (int i = 1;i <= n1;i++) s += tr[ls(t1[i])].sum_cnt;
        for (int i = 1;i <= n2;i++) s -= tr[ls(t2[i])].sum_cnt;
        if (k <= s) {
            for (int i = 1;i <= n1;i++) t1[i] = ls(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = ls(t2[i]);
            return kth_min(l, mid, k);
        }
        else {
            for (int i = 1;i <= n1;i++) t1[i] = rs(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = rs(t2[i]);
            return kth_min(mid + 1, r, k - s);
        }
    }

    int kth_min_sum(int l, int r, int k) {//第k小值之和
        if (l == r) return k * v[l];
        int mid = l + r >> 1;
        int s = 0;
        for (int i = 1;i <= n1;i++) s += tr[ls(t1[i])].sum_cnt;
        for (int i = 1;i <= n2;i++) s -= tr[ls(t2[i])].sum_cnt;
        if (k <= s) {
            for (int i = 1;i <= n1;i++) t1[i] = ls(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = ls(t2[i]);
            return kth_min_sum(l, mid, k);
        }
        else {
            int ss = 0;
            for (int i = 1;i <= n1;i++) ss += tr[ls(t1[i])].sum_val;
            for (int i = 1;i <= n2;i++) ss -= tr[ls(t2[i])].sum_val;

            for (int i = 1;i <= n1;i++) t1[i] = rs(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = rs(t2[i]);
            return kth_min_sum(mid + 1, r, k - s) + ss;
        }
    }

    int less_equall_x_cnt(int l, int r, int x) {//小于等于k的数量（可以用来查排名）
        int ans = 0;
        if (l == r) {
            for (int i = 1;i <= n1;i++) ans += tr[t1[i]].sum_cnt;
            for (int i = 1;i <= n2;i++) ans -= tr[t2[i]].sum_cnt;
            return ans;
        }
        int mid = l + r >> 1;
        if (x <= mid) {
            for (int i = 1;i <= n1;i++) t1[i] = ls(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = ls(t2[i]);
            ans += less_equall_x_cnt(l, mid, x);
        }
        else {
            for (int i = 1;i <= n1;i++) ans += tr[ls(t1[i])].sum_cnt;
            for (int i = 1;i <= n2;i++) ans -= tr[ls(t2[i])].sum_cnt;

            for (int i = 1;i <= n1;i++) t1[i] = rs(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = rs(t2[i]);
            ans += less_equall_x_cnt(mid + 1, r, x);
        }
        return ans;
    }


    int less_equall_x_sum(int l, int r, int x) {//小于等于k的和
        int ans = 0;
        if (l == r) {
            for (int i = 1;i <= n1;i++) ans += tr[t1[i]].sum_val;
            for (int i = 1;i <= n2;i++) ans -= tr[t2[i]].sum_val;
            return ans;
        }
        int mid = l + r >> 1;
        if (x <= mid) {
            for (int i = 1;i <= n1;i++) t1[i] = ls(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = ls(t2[i]);
            ans += less_equall_x_sum(l, mid, x);
        }
        else {
            for (int i = 1;i <= n1;i++) ans += tr[ls(t1[i])].sum_val;
            for (int i = 1;i <= n2;i++) ans -= tr[ls(t2[i])].sum_val;

            for (int i = 1;i <= n1;i++) t1[i] = rs(t1[i]);
            for (int i = 1;i <= n2;i++) t2[i] = rs(t2[i]);
            ans += less_equall_x_sum(mid + 1, r, x);
        }
        return ans;
    }

    void BIT_modify(int i, int k, int cnt) {
        for (;i <= n;i += lbt(i)) insert(root[i], root[i], 1, tot, k, cnt);//处理需要修改的log棵主席树
    }

    int BIT_Query_kthmin(int l, int r, int ql, int qr, int k) {
        n1 = 0, n2 = 0;
        for (int i = qr;i;i -= lbt(i)) t1[++n1] = root[i];
        for (int i = ql - 1;i;i -= lbt(i)) t2[++n2] = root[i];
        return kth_min(l, r, k);
    }

    int BIT_Query_le_cnt(int l, int r, int ql, int qr, int k) {
        n1 = 0, n2 = 0;
        for (int i = qr;i;i -= lbt(i)) t1[++n1] = root[i];
        for (int i = ql - 1;i;i -= lbt(i)) t2[++n2] = root[i];
        return less_equall_x_cnt(l, r, k);
    }
};

```



### Segment Beats！

```c++
//Segment Beats!: 操作:区间取最值,区间加. 查询:区间最值,区间求和
struct Segtree {
#define ls (x << 1)
#define rs (x << 1 | 1)
    struct Tag {
        int tagadd = 0;
        int tagmax = 0;
        int tagmin = 0;
    };
    struct Info {
        int sz = 0;
        ll sum = 0;//和

        int mx = 0;//区间最大值
        int mx2 = 0;//区间严格次大值
        int mxcnt = 0;//最大值出现次数

        int mn = 0;//区间最小值
        int mn2 = 0;//区间严格次小值
        int mncnt = 0;//最小值出现次数
    };
    struct node {
        Info info;
        Tag tag;
    };
    Info friend operator +(const Info& l, const Info& r) {
        Info res;
        res.sz = l.sz + r.sz;
        res.sum = l.sum + r.sum;

        res.mx = max(l.mx, r.mx);
        if (l.mx == r.mx) {
            res.mx2 = max(l.mx2, r.mx2);
            res.mxcnt = l.mxcnt + r.mxcnt;
        }
        else if (l.mx > r.mx) {
            res.mx2 = max(l.mx2, r.mx);
            res.mxcnt = l.mxcnt;
        }
        else {
            res.mx2 = max(l.mx, r.mx2);
            res.mxcnt = r.mxcnt;
        }

        res.mn = min(l.mn, r.mn);
        if (l.mn == r.mn) {
            res.mn2 = min(l.mn2, r.mn2);
            res.mncnt = l.mncnt + r.mncnt;
        }
        else if (l.mn > r.mn) {
            res.mn2 = min(l.mn, r.mn2);
            res.mncnt = r.mncnt;
        }
        else {
            res.mn2 = min(l.mn2, r.mn);
            res.mncnt = l.mncnt;
        }
        return res;
    }

    int n;
    vector<node> tr;
    Segtree(const vector<int>& a, int n) :n(n) {
        tr.resize(n << 2);
        build(a, 1, 1, n);
    }
    void build(const vector<int>& a, int x, int l, int r) {
        if (l == r) {
            tr[x].info = { 1,a[l],a[l],-inf,1,a[l],inf,1 };
            tr[x].tag = { 0,0,0 };
            return;
        }
        else {
            int mid = l + r >> 1;
            build(a, ls, l, mid);
            build(a, rs, mid + 1, r);
            pushup(x);
        }
    }
    void pushup(int x) {//从下往上更新
        tr[x].info = tr[ls].info + tr[rs].info;
    }

    void Tagadd(int x, int k) {
        tr[x].tag.tagadd += k;
        tr[x].tag.tagmax += k;
        tr[x].tag.tagmin += k;
        if (tr[x].info.mx2 != -inf) tr[x].info.mx2 += k;
        if (tr[x].info.mn2 != inf) tr[x].info.mn2 += k;
        tr[x].info.sum += k * tr[x].info.sz;
    }
    void Tagmax(int x, int k) {
        tr[x].tag.tagmax += k;
        if (tr[x].info.mx == tr[x].info.mn) {
            tr[x].info.mx += k;
            tr[x].info.mn += k;
            tr[x].info.sum = tr[x].info.mx * tr[x].info.sz;
        }
        else {
            tr[x].info.mx += k;
            if (tr[x].info.mxcnt + tr[x].info.mncnt == tr[x].info.sz) tr[x].info.mn2 += k;
            tr[x].info.sum += k * tr[x].info.mxcnt;
        }
    }
    void Tagmin(int x, int k) {
        tr[x].tag.tagmin += k;
        if (tr[x].info.mx == tr[x].info.mn) {
            tr[x].info.mx += k;
            tr[x].info.mn += k;
            tr[x].info.sum = tr[x].info.mx * tr[x].info.sz;
        }
        else {
            tr[x].info.mn += k;
            if (tr[x].info.mxcnt + tr[x].info.mncnt == tr[x].info.sz) tr[x].info.mx2 += k;
            tr[x].info.sum += k * tr[x].info.mncnt;
        }
    }

    void pushdown(int x) {//下传标记
        if (tr[x].tag.tagmax) {
            if (tr[ls].info.mx > tr[rs].info.mx) Tagmax(ls, tr[x].tag.tagmax);
            else if (tr[ls].info.mx < tr[rs].info.mx) Tagmax(rs, tr[x].tag.tagmax);
            else {
                Tagmax(ls, tr[x].tag.tagmax);
                Tagmax(rs, tr[x].tag.tagmax);
            }
            tr[x].tag.tagmax = 0;
        }
        if (tr[x].tag.tagmin) {
            if (tr[ls].info.mn < tr[rs].info.mn) Tagmin(ls, tr[x].tag.tagmin);
            else if (tr[ls].info.mn > tr[rs].info.mn) Tagmin(rs, tr[x].tag.tagmin);
            else {
                Tagmin(ls, tr[x].tag.tagmin);
                Tagmin(rs, tr[x].tag.tagmin);
            }
            tr[x].tag.tagmin = 0;
        }
        if (tr[x].tag.tagadd) {
            Tagmin(ls, tr[x].tag.tagadd);
            Tagmin(rs, tr[x].tag.tagadd);
            tr[x].tag.tagadd = 0;
        }
    }

    //区间加
    void update(int x, int l, int r, int ql, int qr, int k) {
        if (l == ql && r == qr) {
            Tagadd(x, k);
            return;
        }
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) update(ls, l, mid, ql, qr, k);
        else if (mid < ql) update(rs, mid + 1, r, ql, qr, k);
        else {
            update(ls, l, mid, ql, mid, k);
            update(rs, mid + 1, r, mid + 1, qr, k);
        }
        pushup(x);
    }

    //区间取min(a_i,k),把大于k的数变成k.
    void ChkMin(int x, int l, int r, int ql, int qr, int k) {
        if (l > qr || r < ql || tr[x].info.mx <= k) return;
        if (ql <= l && r <= qr && tr[x].info.mx2 < k) {
            Tagmax(x, k - tr[x].info.mx);
            return;
        }
        pushdown(x);
        int mid = l + r >> 1;
        ChkMin(ls, l, mid, ql, qr, k);
        ChkMin(rs, mid + 1, r, ql, qr, k);
        pushup(x);
    }
    //区间取max(a_i,k),把小于k的数变成k
    void ChkMax(int x, int l, int r, int ql, int qr, int k) {
        if (l > qr || r < ql || k <= tr[x].info.mn) return;
        if (ql <= l && r <= qr && k < tr[x].info.mn2) {
            Tagmin(x, k - tr[x].info.mn);
            return;
        }
        pushdown(x);
        int mid = l + r >> 1;
        ChkMax(ls, l, mid, ql, qr, k);
        ChkMax(rs, mid + 1, r, ql, qr, k);
        pushup(x);
    }


    Info query(int x, int l, int r, int ql, int qr) {
        if (l == ql && r == qr) return tr[x].info;
        int mid = l + r >> 1;
        pushdown(x);
        if (qr <= mid) return query(ls, l, mid, ql, qr);
        else if (mid < ql) return query(rs, mid + 1, r, ql, qr);
        else return query(ls, l, mid, ql, mid) + query(rs, mid + 1, r, mid + 1, qr);
    }

};
```



### 珂朵莉树

```c++
struct ODT {
    struct node {
        int l, r;
        mutable int v;
        node(int l, int r = -1, int v = 0) :l(l), r(r), v(v) {}
        bool friend operator<(node a, node b) {
            return a.l < b.l;
        }
    };
    set<node> s;
    void insert(int l, int r, int v) {
        s.insert(node(l, r, v));
    }
    auto split(int pos) {
        auto it = s.lower_bound(node(pos));
        if (it != s.end() && it->l == pos) return it;
        it--;
        int l = it->l, r = it->r, v = it->v;
        s.erase(it);
        s.insert(node(l, pos - 1, v));
        return s.insert(node(pos, r, v)).first;
    }
    int query(int pos) {
        auto it = split(pos);
        return it->v;
    }
    void assign(int l, int r, int val) {
        auto itr = split(r + 1), itl = split(l);
        s.erase(itl, itr);
        s.insert(node(l, r, val));
    }
    void add(int l, int r, int val) {
        auto itr = split(r + 1), itl = split(l);
        for (auto it = itl;it != itr;it++) it->v += val;
    }
    int kth(int l, int r, int k) {
        vector<pair<int, int>> vp;
        auto itr = split(r + 1), itl = split(l);
        for (auto it = itl;it != itr;it++) vp.push_back({ it->v, (it->r) - (it->l) + 1 });
        sort(vp.begin(), vp.end());
        for (auto it = vp.begin();it != vp.end();it++) {
            k -= it->second;
            if (k <= 0) return it->first;
        }
        return -1ll;
    }
    int sum(int l, int r, int ex, int mod = MOD) {
        auto itr = split(r + 1), itl = split(l);
        int res = 0;
        for (auto it = itl;it != itr;it++) {
            res = (res + (it->r - it->l + 1) * qp(it->v, ex, mod) % mod) % mod;
        }
        return res;
    }
};
```



### 树链剖分

```c++
struct HLD {
    int idx = 0;
    vector<int> top, dep, fa, sz, son, in, out, dfn, w, nw;
    vector<vector<int>> g;
    HLD(int n) : top(n + 1), dep(n + 1), fa(n + 1), sz(n + 1), son(n + 1),
        in(n + 1), out(n + 1), dfn(n + 1), w(n + 1), nw(n + 1), g(n + 1), idx(0) {}


    void build(int root) {
        dfs(root, root);
        dfs2(root, root, idx);
    }

    void AddEdge(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    void dfs(int u, int p) {
        fa[u] = p;
        dep[u] = dep[p] + 1;
        sz[u] = 1;
        for (int v : g[u]) {
            if (v == p) continue;
            dfs(v, u);
            sz[u] += sz[v];
            //w[v]=g[u].w;//边权的情况，记录为儿子的点权
            if (sz[son[u]] < sz[v]) son[u] = v;//重儿子
        }
    }

    void dfs2(int u, int Top, int& idx) {
        top[u] = Top;
        in[u] = ++idx;
        dfn[idx] = u;
        nw[idx] = w[u];
        if (son[u] == 0) return;
        dfs2(son[u], Top, idx);
        for (int v : g[u]) {
            if (v == fa[u] || v == son[u]) continue;
            dfs2(v, v, idx);
        }
        out[u] = idx;
    }

    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] < dep[top[v]]) swap(u, v);
            u = fa[top[u]];
        }
        return dep[u] < dep[v] ? u : v;
    }

    // 操作路径上的节点
    void operatePath(int a, int b, function<void(int, int)> op, bool isEdge) {
        while (top[a] != top[b]) {
            if (dep[top[a]] < dep[top[b]]) swap(a, b);
            op(in[top[a]], in[a]);
            a = fa[top[a]];
        }
        if (dep[a] > dep[b]) swap(a, b);
        op(in[a] + isEdge, in[b]);
    }

    // 操作子树
    void operateSubtree(int x, function<void(int, int)> op, bool isEdge) {
        op(in[x] + isEdge, in[x] + sz[x] - 1);
    }

    // 获取两点间的距离
    int distance(int a, int b) {
        return dep[a] + dep[b] - 2 * dep[lca(a, b)];
    }

    // 判断节点b是否在节点a的子树中
    bool isAncestor(int a, int b) {
        return in[a] <= in[b] && in[b] <= out[a];
    }

    // 获取三个节点的LCA
    int rootedLca(int a, int b, int c) {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }

    // 从节点u出发，沿着父节点往上跳k层，返回跳到的节点 (k级祖先)
    int kthAncestor(int u, int k) {
        if (dep[u] < k) return -1;
        int d = dep[u] - k;
        while (dep[top[u]] > d) u = fa[top[u]];
        return dfn[in[u] - dep[u] + d];
    }

    // //非交换情况下查询最短路径上的值(目前遇到只用于求树上两点的连续子段最大最小)
    // void opPathNonCommutative(int u, int v, function<void(int, int)> op, function<void(node)> rev) {
    //     node left, tight;
    //     while (top[u] != top[v]) {
    //         if (dep[top[u]] < dep[top[v]]) {
    //             swap(u, v);
    //             swap(left, right);
    //         }
    //         left = op(in[top[u]], in[u]) + left;
    //         u = fa[top[u]];
    //     }
    //     if (dep[u] > dep[v]) {
    //         swap(u, v);
    //         swap(left, right);
    //     }
    //     return rev(left) + op(in[u], in[v]) + right;
    // }


};
```



### 无旋Treap

```c++
mt19937 rng(random_device{}());
//mt19937 rng(99999989);
template <typename T> struct Treap {
#define ls(u) (tr[u].ls)
#define rs(u) (tr[u].rs)
    struct node {
        //origin
        int ls, rs, key, sz;
        T val;
        //extend
        bool tag;//区间翻转标记
        //int fa;//父节点,一般用于维护序列中查找指定节点的中序值.
    };

    static node tr[N];
    inline static int stk[N];
    int root, T1, T2, T3;
    inline static int Top, idx;

    //O(n) build a Treap needed.
    inline static int BUILD_STACK[N], TOP;

    Treap() {
        root = T1 = T2 = T3 = 0;
    }
    ~Treap() {
        for (int i = 1;i <= idx;i++) {
            tr[i] = { 0,0,0,0,0,0 };
        }
        root = idx = Top = 0;
    }
    int BUILD(const vector<T>& vec) {//O(n) build a Treap.
        for (auto i : vec) {
            int id = newnode(i), lst = 0;
            while (TOP && tr[BUILD_STACK[TOP]].key > tr[id].key) {//笛卡尔树建树
                pushup(BUILD_STACK[TOP]);
                lst = BUILD_STACK[TOP--];
            }
            if (TOP) rs(BUILD_STACK[TOP]) = id;
            ls(id) = lst;
            BUILD_STACK[++TOP] = id;
        }
        while (TOP) pushup(BUILD_STACK[TOP--]);
        return BUILD_STACK[1]; //树根
    }

    node operator[](int i) const { return tr[i]; }
    int size() { return tr[root].sz; }//总大小
    int newnode(T val) {
        int nn = Top ? stk[Top--] : ++idx;
        tr[nn] = { 0,0, (int)rng(), 1,val,0 };
        return nn;
    }
    void pushup(int u) {
        tr[u].sz = tr[ls(u)].sz + tr[rs(u)].sz + 1;
        //维护父节点
        /*
        if (ls(u)) tr[ls(u)].fa = u;
        if (rs(u)) tr[rs(u)].fa = u;
        if (ls(tr[u].fa) != u && rs(tr[u].fa) != u) tr[u].fa = 0;
        */
    }

    int merge(int u, int v) {
        if (!u || !v) return u + v;
        if (tr[u].key > tr[v].key) {
            pushdown(u);
            rs(u) = merge(rs(u), v);
            pushup(u);
            return u;
        }
        else {
            pushdown(v);
            ls(v) = merge(u, ls(v));
            pushup(v);
            return v;
        }
    }


    //序列操作

    void pushdown(int u) {
        if (tr[u].tag) {
            if (ls(u)) OP_REV(ls(u));
            if (rs(u)) OP_REV(rs(u));
            tr[u].tag = 0;
        }
    }

    //按排名分裂
    void split_order(int u, int k, int& x, int& y) {
        if (!u) {
            x = y = 0;
            return;
        }
        pushdown(u);
        int s = tr[ls(u)].sz + 1;
        if (k < s) {
            y = u;
            split_order(ls(u), k, x, ls(u));
        }
        else {
            x = u;
            split_order(rs(u), k - s, rs(u), y);
        }
        pushup(u);
    }

    //维护序列时的插入,选择插入到哪个位置
    void insert_order(T val, int k) {
        split_order(root, k, T1, T2);
        root = merge(merge(T1, newnode(val)), T2);
    }
    //在某位置插入一段序列(插入了一整棵树),复杂度O(size(vec))
    void INSERT_order(int k, const vector<T>& vec) {
        split_order(root, k, T1, T2);
        root = merge(merge(T1, BUILD(vec)), T2);
    }
    //插入到序列末尾
    void push_back(T val) {
        root = merge(root, newnode(val));
    }

    //删除指定位置[l,r]的元素
    void erase_order(int l, int r) {
        split_order(root, l - 1, T1, T2);
        split_order(T2, r - l + 1, T2, T3);
        auto del = [&](auto&& del, int u) {
            if (!u) return;
            stk[++Top] = u;
            if (ls(u)) del(del, ls(u));
            if (rs(u)) del(del, rs(u));
            };
        del(del, T2);
        root = merge(T1, T3);
    }

    //查找中序遍历第k个元素
    T kth_order(int k) {
        split_order(root, k - 1, T1, T2);
        split_order(T2, 1, T2, T3);
        T res = tr[T2].val;
        root = merge(T1, merge(T2, T3));
        return res;
    }
    /*
    //查找节点是中序遍历的第几个元素
    int rank_order(int u) {
        int res = tr[ls(u)].sz + 1;
        while (tr[u].fa) {
            if (rs(tr[u].fa) == u) res += tr[ls(tr[u].fa)].sz + 1;
            u = tr[u].fa;
        }
        return res;
    }
    */
    //下传翻转标记
    void OP_REV(int u) {
        swap(ls(u), rs(u));
        tr[u].tag ^= 1;
    }
    //区间翻转
    void reverse(int l, int r) {
        split_order(root, l - 1, T1, T2);
        split_order(T2, r - l + 1, T2, T3);
        OP_REV(T2);
        root = merge(T1, merge(T2, T3));
    }

    //集合操作

    //按值分裂
    void split(int u, T val, int& x, int& y) {
        if (!u) {
            x = y = 0;
            return;
        }
        if (tr[u].val > val) {
            y = u;
            split(ls(u), val, x, ls(u));
        }
        else if (tr[u].val <= val) {
            x = u;
            split(rs(u), val, rs(u), y);
        }
        pushup(u);
    }

    //维护集合时的插入
    void insert(T val) {
        split(root, val, T1, T2);
        root = merge(merge(T1, newnode(val)), T2);
    }

    //删除
    void remove(T val) {
        split(root, val, T1, T2);
        split(T1, val - 1, T1, T3);
        T3 = merge(ls(T3), rs(T3));
        root = merge(merge(T1, T3), T2);
    }
    //带垃圾回收的删除
    void erase(T val) {
        split(root, val, T1, T2);
        split(T1, val - 1, T1, T3);
        if (T3) {
            if (Top < (N >> 8) - 5) stk[++Top] = T3;
        }
        T3 = merge(ls(T3), rs(T3));
        root = merge(merge(T1, T3), T2);
    }
    //小于val的个数+1
    int rank(T val) {
        split(root, val - 1, T1, T2);
        int res = tr[T1].sz + 1;
        root = merge(T1, T2);
        return res;
    }
    //第k小
    T kth(int k) {
        int u = root;
        while (u) {
            int s = tr[ls(u)].sz + 1;
            if (s == k) break;
            else if (k < s) u = ls(u);
            else k -= s, u = rs(u);
        }
        return tr[u].val;
    }
    //val的前驱
    T find_pre(T val) {
        split(root, val - 1, T1, T2);
        int u = T1;
        while (rs(u)) u = rs(u);
        root = merge(T1, T2);
        return tr[u].val;
    }
    //val的后继
    T find_next(T val) {
        split(root, val, T1, T2);
        int u = T2;
        while (ls(u)) u = ls(u);
        root = merge(T1, T2);
        return tr[u].val;
    }

    //l<=val<=r的数量
    int count(T l, T r) {
        split(root, r, T1, T2);
        split(T1, l - 1, T1, T3);
        int res = tr[T3].sz;
        root = merge(merge(T1, T3), T2);
        return res;
    }

    void join(int x, int y) {
        auto run = [&](auto&& run, int x, int y) {
            if (!x || !y) return x + y;
            if (tr[x].key < tr[y].key) swap(x, y);
            int L1 = ls(x), R1 = rs(x), L2 = 0, R2 = 0;
            split(y, tr[x].val, L2, R2), split(L2, tr[x].val - 1, L2, T1);
            if (T1) {
                tr[x].sz += tr[T1].sz;
                //sum,cnt...
            }
            ls(x) = run(run, L1, L2), rs(x) = run(run, R1, R2);
            pushup(x);
            return x;
            };
        root = run(run, x, y);
    }

    vector<T> elements() {
        vector<T> res;
        auto dfs = [&](auto&& dfs, int u)->void {
            if (!u) return;
            pushdown(u);
            dfs(dfs, ls(u));
            res.push_back(tr[u].val);
            dfs(dfs, rs(u));
            };
        dfs(dfs, root);
        return res;
    }

#undef ls
#undef rs 
};
template<typename T> typename Treap<T>::node Treap<T>::tr[N];
//基本功能是维护集合(set),也可以用于维护序列(vector). 
```

### Splay

#### 维护集合

```c++
template <typename T>
class Splay
{
#define ls(x) t[x].ch[0]
#define rs(x) t[x].ch[1]
    const inline static int inf = numeric_limits<T>::max() / 2;
    const inline static int N = 2e6 + 5;

private:
    int rt = 0, Size = 0, Diff = 0;
    inline static int cnt, Top;
    struct Node
    {
        T w;
        int ch[2], fa, tot, sz;
        void init(int _sz, int _tot, int _fa, T _w) { tot = _tot, sz = _sz, fa = _fa, w = _w, ch[0] = ch[1] = 0; }
        void clear() { tot = 0, sz = 0, fa = 0, w = T(), ch[0] = ch[1] = 0; }
    };
    inline static int stk[N];
    inline static Node t[N];
    void pushup(int x)
    {
        t[x].sz = t[ls(x)].sz + t[rs(x)].sz + t[x].tot;
    }
    void rotate(int x)
    {
        int f = t[x].fa, ff = t[f].fa, c = (rs(f) == x), d = t[x].ch[c ^ 1];
        t[x].fa = ff;
        if (ff)
            t[ff].ch[rs(ff) == f] = x;
        t[f].fa = x, t[x].ch[c ^ 1] = f, t[f].ch[c] = d, t[d].fa = f;
        pushup(f), pushup(x);
    }
    void splay(int x, int goal)
    {
        while (t[x].fa != goal)
        {
            int f = t[x].fa, ff = t[f].fa;
            if (ff != goal)
                ((rs(ff) == f) ^ (rs(f) == x)) ? rotate(x) : rotate(f);
            rotate(x);
        }
        if (!goal)
            rt = x;
    }

public: // 返回数字相当于指针
    Splay()
    {
        rt = Size = Diff = 0;
        insert(-inf);
        insert(inf);
    }
    ~Splay()
    {
        for (int i = 1; i <= cnt; i++)
            t[i].clear();
        cnt = Top = 0;
    }
    static T real(int x)
    {
        return t[x].w;
    }
    int find(T x)
    {
        int u = rt;
        while (t[u].w != x && t[u].ch[t[u].w < x])
            u = t[u].ch[t[u].w < x];
        splay(u, 0);
        return u;
    }
    void insert(T w)
    {
        Size++;
        int u = rt, fa = 0;
        while (u && t[u].w != w)
            fa = u, u = t[u].ch[t[u].w < w];
        if (u)
            return ++t[u].tot, ++t[u].sz, splay(u, 0);
        Diff++;
        int newnode = (Top ? stk[Top--] : ++cnt);
        t[newnode].init(1, 1, fa, w);
        if (fa)
            t[fa].ch[t[fa].w < w] = newnode;
        splay(newnode, 0);
    }
    int size()
    {
        return Size - 2;
    }
    int diff()
    {
        return Diff - 2;
    }
    bool empty()
    {
        return (Size - 2);
    }
    int rank(T w)
    {
        int u = rt;
        while (1)
        {
            if (t[ls(u)].sz >= w)
                u = ls(u);
            else if (t[ls(u)].sz + t[u].tot >= w)
            {
                splay(u, 0);
                return u;
            }
            else
                w -= (t[ls(u)].sz + t[u].tot), u = rs(u);
        }
    }
    int count(T x)
    {
        find(x);
        if (t[rt].w != x)
            return 0;
        else
            return t[rt].tot;
    }
    int upper_bound(T x, int opt) // 返回指针，要用_S(x)解除引用
    {                             // opt == 0 前驱，opt == 1 后继
        find(x);
        if (opt == 1 && t[rt].w > x)
            return rt;
        if (opt == 0 && t[rt].w < x)
            return rt;
        int u = t[rt].ch[opt];
        while (t[u].ch[opt ^ 1])
            u = t[u].ch[opt ^ 1];
        splay(u, 0);
        return u;
    }
    int lower_bound(T x, int opt)
    { // opt == 0 前驱，opt == 1 后继 可相同
        find(x);
        if (opt == 1 && t[rt].w >= x)
            return rt;
        if (opt == 0 && t[rt].w <= x)
            return rt;
        int u = t[rt].ch[opt];
        while (t[u].ch[opt ^ 1])
            u = t[u].ch[opt ^ 1];
        return u;
    }

    int order_of_key(T x)
    { // 比x小的数的个数
        int ans = 0;
        insert(x), find(x), ans = t[ls(rt)].sz, erase(x);
        // cout << ans << endl;
        return ans;
    }
    bool erase(T x)
    {
        if (!count(x))
            return false;
        Size--;
        int las = upper_bound(x, 0), nxt = upper_bound(x, 1);
        splay(las, 0), splay(nxt, las);
        int u = t[nxt].ch[0];
        if (t[u].tot > 1)
            --t[u].tot, --t[u].sz, splay(u, 0);
        else
            t[nxt].ch[0] = 0, t[u].clear(), stk[++Top] = u, Diff--; // 清理入队
        return true;
    }
    void clear()
    { // 先遍历子树并删除
        auto dfs = [&](int u, auto dfs) -> void
        {
            if (u == 0)
                return;
            dfs(t[u].ch[0], dfs);
            dfs(t[u].ch[1], dfs);
            t[u].clear(), stk[++Top] = u, Diff--;
        };
        dfs(rt, dfs);
    }
    vector<T> get()
    { // 获取中序遍历数组
        vector<T> v;
        auto dfs = [&](int u, auto dfs) -> void
        {
            if (u == 0)
                return;
            dfs(t[u].ch[0], dfs);
            if (t[u].w != inf && t[u].w != -inf)
                v.push_back(t[u].w);
            dfs(t[u].ch[1], dfs);
        };
        dfs(rt, dfs);
        return v;
    }
};
#define _S(x) Splay<int>::real(x) // 解引用
```



#### 维护序列

```c++
template <typename T>
class Splay // 序列或字符串维护专用
{
#define ls(x) t[x].ch[0]
#define rs(x) t[x].ch[1]
    const inline static int inf = numeric_limits<T>::max() / 2;
    const inline static int N = 1e6 + 5;

private:
    int rt = 0, head, tail, Size;
    inline static int cnt = 0, Top = 0;
    struct Node
    {
        T w, sum;
        int id;                  // 在这里id是下标
        int ch[2], fa, sz, mark; // 左右儿子，父节点，子树大小，标记
        void init(int _sz, int _fa, int _id, T _w) { sz = _sz, fa = _fa, id = _id, sum = w = _w, ch[0] = ch[1] = mark = 0; }
        void clear()
        {
            sz = 0, fa = 0, id = 0, sum = w = T();
            ch[0] = ch[1] = mark = 0;
        }
    };
    inline static int stk[N];
    inline static Node t[N];
    void push_up(int x)
    {
        t[x].sz = t[ls(x)].sz + t[rs(x)].sz + 1;
        t[x].sum = t[ls(x)].sum + t[rs(x)].sum + t[x].w;
    }
    void push_down(int x) // 下传旋转懒标记,可更改成其他区间操作
    {
        if (t[x].mark)
        {
            t[t[x].ch[0]].mark ^= 1;
            t[t[x].ch[1]].mark ^= 1;
            t[x].mark ^= 1;
            swap(t[x].ch[0], t[x].ch[1]);
        }
    }
    void rotate(int x)
    {
        int f = t[x].fa, ff = t[f].fa, c = (rs(f) == x), d = t[x].ch[c ^ 1];
        t[x].fa = ff;
        if (ff)
            t[ff].ch[rs(ff) == f] = x;
        t[f].fa = x, t[x].ch[c ^ 1] = f, t[f].ch[c] = d, t[d].fa = f;
        push_up(f), push_up(x);
    }
    void splay(int x, int goal)
    {
        while (t[x].fa != goal)
        {
            int f = t[x].fa, ff = t[f].fa;
            if (ff != goal)
                ((rs(ff) == f) ^ (rs(f) == x)) ? rotate(x) : rotate(f);
            rotate(x);
        }
        if (!goal)
            rt = x;
    }
    int find(int x)
    {
        int u = rt;
        x++; // 一开始有个哨兵节点
        while (1)
        {
            push_down(u);
            if (x <= t[t[u].ch[0]].sz)
            {
                u = t[u].ch[0];
            }
            else
            {
                int tmp = (t[u].ch[0] ? t[t[u].ch[0]].sz : 0) + 1;
                if (x == tmp)
                    return u;
                else
                    x -= tmp, u = t[u].ch[1];
            }
        }
        // splay(u, 0);//这里只返回指针，不splay
    }
    void insert(int id, T w)
    {
        Size++;
        int u = rt, fa = 0;
        while (u && t[u].id != id)
            fa = u, u = t[u].ch[t[u].id < id];
        int newnode = (Top ? stk[Top--] : ++cnt);
        t[newnode].init(1, fa, id, w);
        if (fa)
            t[fa].ch[t[fa].id < id] = newnode;
        splay(newnode, 0);
    }

public:           // 返回数字相当于指针
    Splay(int sz) // 开数组,范围1-n
    {
        rt = 0;
        Size = sz;
        head = 0, tail = sz + 1;
        insert(head, T());
        insert(tail, T()); // 插入哨兵节点
        for (int i = 1; i <= sz; i++)
            insert(i, T());
    }
    ~Splay()
    {
        clear();
    }
    static T real(int x)
    {
        return t[x].id;
    }

    T get(int id)
    {
        int pos = find(id);
        splay(pos, 0);
        return t[pos].w;
    }
    void set(int id, T w) // 单点操作
    {
        int pos = find(id);
        t[pos].w = w;
        splay(pos, 0);
    }
    void add(int id, T w) // 单点操作
    {
        int pos = find(id);
        // cout << t[pos].w << endl;
        t[pos].w += w;
        // cout << t[pos].w << endl;
        splay(pos, 0);
    }
    T get_range(int l, int r)
    {
        if (l < 1 || r > Size)
            return T();
        int ll = find(l - 1), rr = find(r + 1);
        splay(ll, 0);
        splay(rr, ll);
        return t[t[t[rt].ch[1]].ch[0]].sum;
    }

    void clear()
    { // 先遍历子树并删除
        auto dfs = [&](int u, auto dfs) -> void
        {
            if (u == 0)
                return;
            dfs(t[u].ch[0], dfs);
            dfs(t[u].ch[1], dfs);
            t[u].clear(), stk[++Top] = u;
        };
        dfs(rt, dfs);
        Size = 0;
    }
    vector<pair<int, T>> get_all() // 获取全部的下标和值
    {                              // 获取中序遍历数组和数量
        vector<pair<int, T>> v;
        auto dfs = [&](int u, auto dfs) -> void
        {
            if (u == 0)
                return;
            push_down(u);
            dfs(t[u].ch[0], dfs);
            if (t[u].id != head && t[u].id != tail)
                v.push_back({t[u].id, t[u].w});
            dfs(t[u].ch[1], dfs);
        };
        dfs(rt, dfs);
        return v;
    }
    void reverse(int l, int r)
    { //
        int ll = find(l - 1), rr = find(r + 1);
        // cout << t[ll].sz << endl;
        splay(ll, 0);
        splay(rr, ll);
        t[t[t[rt].ch[1]].ch[0]].mark ^= 1;
    }
};
#define _S(x) Splay<int>::real(x) // 解引用
```

### LCT

```c++
#include<bits/stdc++.h>
#define N 300005
using namespace std;
int n,m,val[N];
struct Link_Cut_Tree{
    int top,c[N][2],fa[N],xr[N],q[N],rev[N];
    inline void pushup(int x){xr[x]=xr[c[x][0]]^xr[c[x][1]]^val[x];}
    inline void pushdown(int x){
        int l=c[x][0],r=c[x][1];
        if(rev[x]){
            rev[l]^=1;rev[r]^=1;rev[x]^=1;
            swap(c[x][0],c[x][1]);
        }
    }
    inline bool isroot(int x){return c[fa[x]][0]!=x&&c[fa[x]][1]!=x;}
    void rotate(int x){
        int y=fa[x],z=fa[y],l,r;
        if(c[y][0]==x)l=0;else l=1;r=l^1;
        if(!isroot(y)){if(c[z][0]==y)c[z][0]=x;else c[z][1]=x;}
        fa[x]=z;fa[y]=x;fa[c[x][r]]=y;
        c[y][l]=c[x][r];c[x][r]=y;
        pushup(y);pushup(x);
    }
    void splay(int x){
        top=1;q[top]=x;
        for(int i=x;!isroot(i);i=fa[i])q[++top]=fa[i];
        for(int i=top;i;i--)pushdown(q[i]);
        while(!isroot(x)){
            int y=fa[x],z=fa[y];
            if(!isroot(y)){
                if((c[y][0]==x)^(c[z][0]==y))rotate(x);
                else rotate(y);
            }rotate(x);
        }
    }
    void access(int x){for(int t=0;x;t=x,x=fa[x])splay(x),c[x][1]=t,pushup(x);}
    void makeroot(int x){access(x);splay(x);rev[x]^=1;}
    int find(int x){access(x);splay(x);while(c[x][0])x=c[x][0];return x;}
    void split(int x,int y){makeroot(x);access(y);splay(y);}
    void cut(int x,int y){split(x,y);if(c[y][0]==x&&c[x][1]==0)c[y][0]=0,fa[x]=0;}
    void link(int x,int y){makeroot(x);fa[x]=y;}
}T;
inline int read(){
    int f=1,x=0;char ch;
    do{ch=getchar();if(ch=='-')f=-1;}while(ch<'0'||ch>'9');
    do{x=x*10+ch-'0';ch=getchar();}while(ch>='0'&&ch<='9');
    return f*x;
}
int main(){
    n=read();m=read();
    for(int i=1;i<=n;i++)val[i]=read(),T.xr[i]=val[i];
    while(m--){
        int opt=read();
        if(opt==0){
            int x=read(),y=read();T.split(x,y);
            printf("%d\n",T.xr[y]);
        }
        if(opt==1){
            int x=read(),y=read(),xx=T.find(x),yy=T.find(y);
            if(xx!=yy)T.link(x,y);
        }
        if(opt==2){
            int x=read(),y=read(),xx=T.find(x),yy=T.find(y);
            if(xx==yy)T.cut(x,y);
        }
        if(opt==3){
            int x=read(),y=read();
            T.access(x);T.splay(x);val[x]=y;T.pushup(x);
        }
    }
    return 0;
}
```

### LCT维护树的重心

```cpp
#define R register int
#define I inline void
const int N=100009,INF=2147483647;
int f[N],c[N][2],si[N],s[N],h[N];
bool r[N];
#define lc c[x][0]
#define rc c[x][1]
inline bool nroot(R x){return c[f[x]][0]==x||c[f[x]][1]==x;}
I pushup(R x){
    s[x]=s[lc]+s[rc]+si[x]+1;
}
I pushdown(R x){
    if(r[x]){
        R t=lc;lc=rc;rc=t;
        r[lc]^=1;r[rc]^=1;r[x]=0;
    }
}
I pushall(R x){
    if(nroot(x))pushall(f[x]);
    pushdown(x);
}
I rotate(R x){
    R y=f[x],z=f[y],k=c[y][1]==x,w=c[x][!k];
    if(nroot(y))c[z][c[z][1]==y]=x;
    f[f[f[c[c[x][!k]=y][k]=w]=y]=x]=z;pushup(y);//为三行rotate打call
}
I splay(R x){
    pushall(x);
    R y;
    while(nroot(x)){
    	if(nroot(y=f[x]))rotate((c[f[y]][0]==y)^(c[y][0]==x)?x:y);
    	rotate(x);
    }
    pushup(x);
}
I access(R x){
    for(R y=0;x;x=f[y=x]){
        splay(x);
        si[x]+=s[rc];
        si[x]-=s[rc=y];
        pushup(x);
    }
}
I makeroot(R x){
    access(x);splay(x);
    r[x]^=1;
}
I split(R x,R y){
    makeroot(x);
    access(y);splay(y);
}
I link(R x,R y){
    split(x,y);
    si[f[x]=y]+=s[x];
    pushup(y);
}
int geth(R x){
    if(h[x]==x)return x;
    return h[x]=geth(h[x]);
}
inline int update(R x){
    R l,r,ji=s[x]&1,sum=s[x]>>1,lsum=0,rsum=0,newp=INF,nowl,nowr;
    while(x){
        pushdown(x);//注意pushdown
        nowl=s[l=lc]+lsum;nowr=s[r=rc]+rsum;
        if(nowl<=sum&&nowr<=sum){
            if(ji){newp=x;break;}//剪枝，确定已经直接找到
            else if(newp>x)newp=x;//选编号最小的
        }
        if(nowl<nowr)lsum+=s[l]+si[x]+1,x=r;
        else         rsum+=s[r]+si[x]+1,x=l;//缩小搜索区间
    }
    splay(newp);//保证复杂度
    return newp;
}
#define G ch=getchar()
#define gc G;while(ch<'-')G
#define in(z) gc;z=ch&15;G;while(ch>'-')z*=10,z+=ch&15,G;
int main(){
    register char ch;
    R n,m,x,y,z,Xor=0;
    in(n);in(m);
    for(R i=1;i<=n;++i)s[i]=1,h[i]=i,Xor^=i;
    while(m--){
    	gc;
    	switch(ch){
    		case 'A':in(x);in(y);link(x,y);
    			split(x=geth(x),y=geth(y));//提出原重心路径
    			z=update(y);
    			Xor=Xor^x^y^z;
    			h[x]=h[y]=h[z]=z;//并查集维护好
    			break;
    		case 'Q':in(x);printf("%d\n",geth(x));break;
    		case 'X':gc;gc;printf("%d\n",Xor);
    	}
    }
    return 0;
}
```

### LCT维护区间点权最大值(维护最小生成树)

``` cpp
int va[N];

namespace LCT
{
	// origin
	int ch[N][2], fa[N], stk[N], rev[N];
#define ls(x) ch[x][0]
#define rs(x) ch[x][1]
	// extend
	int val[N], sum[N];
	//extend
	int mx[N];
	// extend
	int tag[N], siz[N]; // 区间推平,链长
	void init(int n)
	{ // 初始化link-cut-tree
		for (int i = 0; i <= n; i++)
			ch[i][0] = ch[i][1] = 0;
		for (int i = 0; i <= n; i++)
			fa[i] = 0;
		for (int i = 0; i <= n; i++)
			val[i] = sum[i] = 0;
		for (int i = 0; i <= n; i++)
			tag[i] = 0;
		for (int i = 0; i <= n; i++)
			siz[i] = 0;
	}
	inline bool son(int x)
	{
		return ch[fa[x]][1] == x;
	}
	inline bool isroot(int x)
	{
		return ch[fa[x]][1] != x && ch[fa[x]][0] != x;
	}
	inline void reverse(int x)
	{ // 给结点x打上反转标记
		swap(ch[x][1], ch[x][0]);
		rev[x] ^= 1;
	}
	void cao(int x, int y)
	{ // 区间赋值操作
		val[x] = tag[x] = y;
		sum[x] = y * siz[x];
	}
	inline void pushup(int x)
	{
		sum[x] = sum[ls(x)] + sum[rs(x)] + val[x];
		siz[x] = siz[ls(x)] + siz[rs(x)] + 1;
		mx[x]=val[x];
		if(e[mx[ls(x)]].z>e[mx[x]].z)mx[x]=mx[ls(x)];
		if(e[mx[rs(x)]].z>e[mx[x]].z)mx[x]=mx[rs(x)];
	}

	inline void pushdown(int x)
	{
		if (tag[x] != -1)
		{
			if (ls(x))
				cao(ls(x), tag[x]);
			if (rs(x))
				cao(rs(x), tag[x]);
			tag[x] = -1;
		}
		if (rev[x])
		{
			reverse(ls(x));
			reverse(rs(x));
			rev[x] = 0;
		}
	}
	void rotate(int x)
	{
		int y = fa[x], z = fa[y], c = son(x);
		if (!isroot(y))
			ch[z][son(y)] = x;
		fa[x] = z;
		ch[y][c] = ch[x][!c];
		fa[ch[y][c]] = y;
		ch[x][!c] = y;
		fa[y] = x;
		pushup(y);
	}
	void splay(int x)
	{
		int top = 0;
		stk[++top] = x;
		for (int i = x; !isroot(i); i = fa[i])
			stk[++top] = fa[i];
		while (top)
			pushdown(stk[top--]);
		for (int y = fa[x]; !isroot(x); rotate(x), y = fa[x])
			if (!isroot(y))
				son(x) ^ son(y) ? rotate(x) : rotate(y);
		pushup(x);
	}
	void access(int x)
	{
		for (int y = 0; x; y = x, x = fa[x])
		{
			splay(x);
			ch[x][1] = y;
			pushup(x);
		}
	}
	void makeroot(int x)
	{ // 将x变为树的新的根结点
		access(x);
		splay(x);
		reverse(x);
	}
	int findroot(int x)
	{ // 返回x所在树的根结点
		access(x);
		splay(x);
		while (ch[x][0])
			pushdown(x), x = ch[x][0];
		return x;
	}
	void split(int x, int y)
	{ // 提取出来y到x之间的路径，并将y作为根结点
		makeroot(x);
		access(y);
		splay(y);
	}
	void cut(int x)
	{ // 断开结点x与它的父结点之间的边
		access(x);
		splay(x);
		ch[x][0] = fa[ls(x)] = 0;
		pushup(x);
	}
	bool sametree(int x, int y)
	{ // 判断结点x与y是否属于同一棵树
		makeroot(x);
		return findroot(y) == x;
	}
	void cut(int x, int y)
	{ // 切断x与y相连的边(必须保证x与y在一棵树中)
		if (!sametree(x, y))
			return;
		makeroot(x); // 将x置为整棵树的根
		if (fa[y] == x)
			cut(y); // 删除y与其父结点之间的边
	}
	void link(int x, int y)
	{ // 连接x与y(必须保证x和y属于不同的树)
		if (sametree(x, y))
			return;
		makeroot(x);
		fa[x] = y;
	}
	void change(int x, int y)
	{ // x节点的值改为y
		splay(x);
		val[x] = y;
		pushup(x);
	}

	int gao(int a, int b, int c, int d)
	{
		split(a, b);
		cao(b, 1);
		split(c, d);
		int res = sum[d];
		split(a, b);
		cao(b, 0);
		return res;
	}
	int modify(int x,int y,int num){//x到y的点权赋值为1
		split(x,y);
		cao(num);
	}
	int get(int x,int y){//获取x到y的路径点权和
		split(x,y);
		return sum[y];
	}

}
using namespace LCT;//未联通直接加边，否则找当前u-v之间最大边尝试替换，记得把边用一共点代替
```

### LCT维护子树权值

```cpp
const int N = 1e5 + 5;


namespace LCT
{
	// origin
	int ch[N][2], fa[N], stk[N], rev[N];
#define ls(x) ch[x][0]
#define rs(x) ch[x][1]
	// extend
	int val[N], sum[N];
	// extend
	int mx[N];
	// extend
	int exsz[N], sz[N];
	inline bool son(int x)
	{
		return ch[fa[x]][1] == x;
	}
	inline bool isroot(int x)
	{
		return ch[fa[x]][1] != x && ch[fa[x]][0] != x;
	}
	inline void reverse(int x)
	{
		swap(ch[x][1], ch[x][0]);
		rev[x] ^= 1;
	}
	inline void pushup(int x)
	{
		sz[x] = sz[ls(x)] + sz[rs(x)] + exsz[x] + val[x];
	}

	inline void pushdown(int x)
	{

		if (rev[x])
		{
			reverse(ls(x));
			reverse(rs(x));
			rev[x] = 0;
		}
	}
	void rotate(int x)
	{
		int y = fa[x], z = fa[y], c = son(x);
		if (!isroot(y))
			ch[z][son(y)] = x;
		fa[x] = z;
		ch[y][c] = ch[x][!c];
		fa[ch[y][c]] = y;
		ch[x][!c] = y;
		fa[y] = x;
		pushup(y);
	}
	void splay(int x)
	{
		int top = 0;
		stk[++top] = x;
		for (int i = x; !isroot(i); i = fa[i])
			stk[++top] = fa[i];
		while (top)
			pushdown(stk[top--]);
		for (int y = fa[x]; !isroot(x); rotate(x), y = fa[x])
			if (!isroot(y))
				son(x) ^ son(y) ? rotate(x) : rotate(y);
		pushup(x);
	}
	void access(int x)
	{
		for (int y = 0; x; y = x, x = fa[x])
		{
			splay(x);
			exsz[x] += sz[rs(x)] - sz[y];//把rs转为y
			rs(x) = y;
			pushup(x);
		}
	}
	void makeroot(int x)
	{ // 将x变为树的新的根结点
		access(x);
		splay(x);
		reverse(x);
	}
	int findroot(int x)
	{ // 返回x所在树的根结点
		access(x);
		splay(x);
		while (ch[x][0])
			pushdown(x), x = ch[x][0];
		return x;
	}
	void split(int x, int y)
	{ // 提取出来y到x之间的路径，并将y作为根结点
		makeroot(x);
		access(y);
		splay(y);
	}
	void cut(int x)
	{ // 断开结点x与它的父结点之间的边
		access(x);
		splay(x);
		ch[x][0] = fa[ls(x)] = 0;
		pushup(x);
	}
	bool sametree(int x, int y)
	{ // 判断结点x与y是否属于同一棵树
		makeroot(x);
		return findroot(y) == x;
	}
	void cut(int x, int y)
	{ // 切断x与y相连的边(必须保证x与y在一棵树中)
		if (!sametree(x, y))
			return;
		makeroot(x); // 将x置为整棵树的根
		if (fa[y] == x)
			cut(y); // 删除y与其父结点之间的边
	}
	void link(int x, int y)
	{ // 连接x与y(必须保证x和y属于不同的树)
		if (sametree(x, y))
			return;
		split(x, y);
		fa[x] = y;
		exsz[y] += sz[x];
		pushup(y);
	}
}
using namespace LCT; // 未联通直接加边，否则找当前u-v之间最大边尝试替换，记得把边用一共点代替

void solve()
{
	int n, q;
	cin >> n >> q;
	for(int i=1;i<=n;i++)val[i]=1;
	for (int i = 1; i <= q; i++)
	{
		char op;
		int x, y;
		cin >> op >> x >> y;
		if (op == 'A')
		{
			link(x, y);
		}
		else
		{
			split(x, y);
			//cout << sz[x] << " " << sz[y] << endl;
			cout << (exsz[x] + val[x]) * (exsz[y] + val[y]) << endl;//左右两个子树权值乘积
		}
	}
}
```



### 点分治

```c++
#include<bits/stdc++.h>
using namespace std;
const int N=10001;
int read() {
    int x=0,f=1;
    char c=getchar();
    while(c<'0'||c>'9')c=='-'?f=-1,c=getchar():c=getchar();
    while(c>='0'&&c<='9') x=x*10+c-'0',c=getchar();
    return x*f;
}
int n,k;
int ans[10000001];/*储存答案*/
int dis[N];/*从当前节点i到枚举当前树的根节点父亲的距离*/(这里随便理解一下吧，我这么说是为了后面的容斥)
int f[N];/*当以i为根节点时最大子树大小*/
int vis[N];/*i节点是否被当根使用过*/
int siz[N];/*以i节点为根时,其子树(包括本身)的节点个数*/
int root;/*根节点*/
int sum;/*这棵当前递归的这棵树的大小*/
struct node {
    int next,to,v;
} a[N<<1];
int head[N],cnt;
void add(int x,int y,int c) {
    a[++cnt].to=y;
    a[cnt].next=head[x];
    a[cnt].v=c;
    head[x]=cnt;
}
void findroot(int k,int fa) {
    f[k]=0,siz[k]=1;
    for(int i=head[k]; i; i=a[i].next) {
        int v=a[i].to;
        if(vis[v]||v==fa)
            continue;
        findroot(v,k);
        siz[k]+=siz[v];
        f[k]=max(f[k],siz[v]);
    }
    f[k]=max(f[k],sum-siz[k]);
    if(f[k]<f[root])
        root=k;
}
int tot;
void finddep(int k,int fa,int l) {
    dis[++tot]=l;
    for(int i=head[k]; i; i=a[i].next) {
        int v=a[i].to;
        if(v==fa||vis[v])
            continue;
        finddep(v,k,l+a[i].v);
    }
}
void calc(int k,int l,int c) {
    tot=0;
    finddep(k,0,l);
    for(int i=1; i<=tot; i++)
        for(int j=1; j<=tot; j++)
            ans[dis[i]+dis[j]]+=c;
}
void devide(int k) {
    vis[k]=1;
    calc(k,0,1);
    for(int i=head[k]; i; i=a[i].next) {
        int v=a[i].to;
        if(vis[v])
            continue;
        calc(v,a[i].v,-1);
        root=0,sum=siz[v];
        findroot(v,0);
        devide(root);
    }
}
int main() {
    int n=read(),m=read(),x,y,z;
    for (int i=1; i<n; i++)
        x=read(),y=read(),z=read(),add(x,y,z),add(y,x,z);
    sum=f[0]=n;
    findroot(1,0);
    devide(root);
    for (int i=1; i<=m; i++) {
        int k=read();
        puts(ans[k]?"AYE":"NAY");
    }
    return 0;
}
```



### 分块

#### 莫队算法

```c++
struct node {
    int l, r;
    int p;
    bool operator<(node a, node b)const {
        if (pos[a.l] == pos[b.l]) {
            return a.r < b.r;
        }
        return pos[a.l] < pos[b.l];
    }
};
void Solve() {
    int n, m, k;
    cin >> n >> m >> k;
    vi a(n + 1, 0);
    for (int i = 1;i <= n;i++) {
        cin >> a[i];
    }
    vc<node> q(m + 1);
    vi pos(n + 1);
    int sz = sqrt(n);
    vi mp(k + 1);
    //分块
    for (int i = 1;i <= n;i++) {
        pos[i] = i / sz;
    }
    for (int i = 1;i <= m;i++) {
        cin >> q[i].l >> q[i].r;
        q[i].p = i;
    }
    sort(q.begin() + 1, q.end(), [&](node a, node b) {
        if (pos[a.l] == pos[b.l]) {
            return a.r < b.r;
        }
        return pos[a.l] < pos[b.l];
        });
    auto add = [&](int x) {
        //具体见题目
        //now...
        };
    auto sub = [&](int x) {
        //具体见题目
        //now...
        };
    int l = 1, r = 0;
    int now = 0;
    vi res(m + 1);
    for (int i = 1;i <= m;i++) {
        while (q[i].l < l) add(--l);
        while (r < q[i].r) add(++r);
        while (q[i].r < r) sub(r--);
        while (l < q[i].l) sub(l++);
        res[q[i].p] = now;
    }
    for (int i = 1;i <= m;i++) {
        cout << res[i] << endl;
    }
}

```

### 带修莫队

```cpp
int l = 0, r = 0, t = 0, nowAns = 0;

inline void move(int pos, int sign) {
    // update nowAns
}

inline void moveTime(int t, int sign) {
    // apply or revoke modification
    // update nowAns
}

void solve() {
    BLOCK_SIZE = int(ceil(pow(n, 2.0 / 3)));
    sort(querys, querys + m);
    for (int i = 0; i < q1; ++i) {
        const query q = querys[i];
        while (t < q.t) moveTime(t++, 1);
        while (t > q.t) moveTime(--t, -1);
        while (l < q.l) move(l++, -1);
        while (l > q.l) move(--l, 1);
        while (r < q.r) move(r++, 1);
        while (r > q.r) move(--r, -1);
        ans[q.id] = nowAns;
    }
}
//加上这个修改：我们首先判断 pos 是否在区间 [l,r] 内。如果是的话，我们等于是从区间中删掉颜色 a，加上颜色 b，并且当前颜色序列的第 pos 项的颜色改成 b。//如果不在区间 [l,r] 内的话，我们就直接修改当前颜色序列的第 pos 项为 b。
//还原这个修改：等于加上一个修改第 pos 项、把颜色 b 改成颜色 a 的修改。
```



#### 高维偏序

```c++

// rank[k][v] 表示 k 维中值为 v 的点的编号 
for (register int k = 0; k < K; k++)
	for (register int i = 1; i * b <= n; i++)
		for (register int j = 1; j <= i * b; j++)
			dat[k][i].set(rank[k][j]); // 分块预处理 

for (register int i = 1; i <= n; i++) {
	bitset<N> ans, tmp;
	ans.set(); // 一开始设为全 1（按位与操作）
	for (register int k = 0; k < K; k++) {
		tmp.reset(); // 每一维都要重置 
		int p = point[k][i] / b; // 计算整块的范围 
		tmp |= dat[k][p]; // 整块取现成 
		for (register int j = p * b + 1; j <= point[k][i]; j++)
			tmp.set(rank[k][j]); // 暴力扫散块 
		ans &= tmp; // 对每一维按位与 
	}
	cout << ans.count() - 1 << endl; // 统计答案 
}
```



### 线性基

```c++
const int B = 60;
struct liner_base {
    vector<ll> num;
    vector<ll> new_num;
    int zero, tot;
    liner_base() {
        num.resize(B + 1);
        new_num.resize(B + 1);
        zero = 0, tot = 0;
    }
    int insert(ll x) {
        for (int i = B;i >= 0;i--) {
            if (x >> i & 1) {
                if (num[i] == 0) {
                    num[i] = x;return 1;
                }
                else x ^= num[i];
            }
        }
        zero = 1;
        return 0;
    }
    //最大
    ll Max() {
        ll res = 0;
        for (int i = B;i >= 0;i--) {
            res = max(res, res ^ num[i]);
        }
        return res;
    }
    //最小
    ll Min() {
        if (zero) return 0;
        for (int i = 0;i <= B;i++) {
            if (num[i]) return num[i];
        }
        return -1;
    }
    //与x异或最大
    ll query_max(ll x) {
        for (int i = B;i >= 0;i--) {
            x = max(x, x ^ num[i]);
        }
        return x;
    }
    //与x异或最小
    ll query_min(ll x) {
        for (int i = B;i >= 0;i--) {
            x = min(x, x ^ num[i]);
        }
        return x;
    }

    //查询x能否被异或出来
    ll ask(int x) {
        for (int i = B;i >= 0;i--) {
            if (x >> i & 1) x ^= num[i];
        }
        return x == 0;
    }
    //重构线性基
    void rebuild() {
        for (int i = B;i >= 0;i--) {
            for (int j = i - 1;j >= 0;j--) {
                if (num[i] >> j & 1) num[i] ^= num[j];
            }
        }
        for (int i = 0;i <= B;i++) {
            if (num[i]) new_num[tot++] = num[i];
        }
    }
    //查询第k小 
    //k - zero ? kth(k - zero) : 0;
    ll kth(int k) {
        if (k >= (1ll << tot)) return -1;
        int res = 0;
        for (int i = B;i >= 0;i--) {
            if (k >> i & 1) res ^= new_num[i];
        }
        return res;
    }
    //排名
    int rank(int x) {
        int res = 0;
        for (int i = tot - 1;i >= 0;i--) {
            if (x >= new_num[i]) res += (1ll << i), x ^= new_num[i];
        }
        return res + zero;
    }
};

```



## 图论

### 按秩合并并查集

```cpp
int fa[100005], sz[100005];
void init(int n)
{
    for (int i = 1; i <= n; i++)
    {
        fa[i] = i;
        sz[i] = 1;
    }
}
int find(int a)
{
    return fa[a] == a ? a : fa[a] = find(fa[a]);
}
void merge(int a, int b)
{
    int aa = find(a), bb = find(b);
    if (aa != bb)
    {
        if (sz[aa] > sz[bb])
        {
            sz[aa] += sz[bb];
            fa[bb] = aa;
        }
        else
        {
            sz[bb] += sz[aa];
            fa[aa] = bb;
        }
    }
}
```

### 点分治

``` cpp
vector<pair<int, int>> pth[500005];
int siz[500005], maxp[500005], vis[500005], root, ans = 0;

void get_root(int pos, int fa, int total)
{
    siz[pos] = 1;
    maxp[pos] = 0;
    for (auto &[to, w] : pth[pos])
    {
        if (to == fa || vis[to])
            continue;
        get_root(to, pos, total);
        siz[pos] += siz[to];
        maxp[pos] = max(maxp[pos], siz[to]);
    }
    maxp[pos] = max(maxp[pos], total - siz[pos]);
    if (!root || maxp[pos] < maxp[root])
        root = pos;
}

void calc(int pos)
{

}
void sol(int pos)
{

    vis[pos] = 1;
    calc(pos);
    for (auto &[to, w] : pth[pos])
    {
        if (vis[to])
            continue;
        maxp[root = 0] = n;
        get_root(to, 0, siz[to]);
        sol(root);
    }
}
```

### 点分树

```cpp
vector<pair<int, int>> pth[500005];//性质：树高logn 原图中的连通块在点分树上也是连通块
int siz[500005], maxp[500005], vis[500005], root, ans = 0,n;
vector<int>pth[500005];
void get_root(int pos, int fa, int total)
{
    siz[pos] = 1;
    maxp[pos] = 0;
    for (auto &[to, w] : pth[pos])
    {
        if (to == fa || vis[to])
            continue;
        get_root(to, pos, total);
        siz[pos] += siz[to];
        maxp[pos] = max(maxp[pos], siz[to]);
    }
    maxp[pos] = max(maxp[pos], total - siz[pos]);
    if (!root || maxp[pos] < maxp[root])
        root = pos;
}
void sol(int pos)
{

    vis[pos] = 1;
    for (auto &[to, w] : pth[pos])
    {
        if (vis[to])
            continue;
        maxp[root = 0] = n;
        get_root(to, 0, siz[to]);
        pth[pos].push_back(root);
        sol(root);
    }
}
void get_tree(int n){
    root=get_root(1,0,n);
    sol(root);
}
```

### 边，点覆盖基础定理

``` cpp
// 最小点覆盖：
// 点覆盖的概念定义：
// 对于图G=(V,E)中的一个点覆盖是一个集合S⊆V使得每一条边至少有一个端点在S中。

// 最小点覆盖：就是中点的个数最少的S集合。
// 普通图的最小点覆盖数好像只能用搜索解，没有什么比较好的方法（可能我比较弱。。）所以在此只讨论二分图的最小点覆盖的求法

// 结论： 二分图的最小点覆盖数=该二分图的最大匹配数，具体证明的方法看大佬博客，里面还给出了如何求具体的最小覆盖的点是哪些点。

// 最小边覆盖：
// 边覆盖的概念定义：
// 边覆盖是图的一个边子集，使该图上每一节点都与这个边子集中的一条边关联，只有含孤立点的图没有边覆盖，边覆盖也称为边覆盖集，图G的最小边覆盖就是指边数最少的覆盖，图G的最小边覆盖的边数称为G的边覆盖数。

// 普通图 的最小边覆盖好像也没有什么除了暴力好的解法，自己菜（逃

// 结论： 二分图的最小边覆盖数=图中的顶点数-（最小点覆盖数）该二分图的最大匹配数

// 最大匹配：
// 匹配：在图论中，一个「匹配」（matching）是一个边的集合，其中任意两条边都没有公共顶点。

// 最大匹配：一个图所有匹配中，所含匹配边数最多的匹配，称为这个图的最大匹配。

// 算法： 匈牙利算法求二分图的最大匹配

// 最小路径覆盖：
// DAG图的最小路径覆盖可以转化为二分图的人后求解直接上大佬的博客吧反正让我讲也讲不出什么花来，我只是整合一下，学起来比较系统。

// 还有无向图的最小路径覆盖，找了很多资料都没有找到合适的解释，这里有一篇博客提到了，但是没有找到其他的资料证明他的正确性，严重的颠覆了我的认知。

// 最大独立集：//常用于找左边哪些点选了，右边就不能选哪些点了
// 最大独立集：在Ｎ个点的图G中选出m个点，使这m个点两两之间没有边的点中，m的最大值。
// 结论： 二分图的最大点独立数=点的个数-最小点覆盖数（最大匹配）
// 证明： 除过最小点覆盖集，剩下的点全部都是互相独立的，因为它们任意两个点之间都没有直接的连边。
// 我们用反证法来证明一下，设最小点覆盖集为V，假如有两个没在V中的点之间有一条边，那么这条边就不会被V中的点所覆盖，那么V就不是
// 最小点覆盖集，又因为V是最小点覆盖集，所以刚才假设的两个点时不存在的，座椅，除过V之外的点都是两两相互独立的。
最小权点覆盖集
一、定义
什么是点覆盖集呢？就是图中所有点的一个子集，首先他是一个点集，然后图中所有边的两个端点的其中一个都在这个点集中，就是说这个点集中包含了所有边的至少一个端点，这个点集就覆盖了所有边。那么对于每个点我们给他一个权值，所有点覆盖集中，总权值和最小的一个就是所说的最小权点覆盖集。

这个问题是一个NP完全问题，就是没有一个更快的算法，只能通过枚举暴搜来实现，但这里我们来看一种特殊的最小权点覆盖集，就是对于一个二分图而言的。

在二分图中，有一个特殊的性质：当所有点权都为1时，最大匹配数 = 最小权点覆盖集；最大权独立集 = n - 最小权点覆盖集。

二、具体做法
我们利用最小割来求解最小权点覆盖集，首先我们是在二分图上来做，如果点权为负数，那么我们直接选，因为选了这个点后，依旧是点覆盖集，而总权值会缩小，所以我们肯定选；而对于所有点权为正数的点，我们用以下方法来解决。我们将所有二分图的点看成两个集合X和Y，从s向所有X集合的点连一条容量为点权的边，从所有Y集合的点向t连一条容量为点权的边（只对于点权为正的点，因为网络流的容量必须是正的），X集合和Y集合之间建原图存在的边，容量为正无穷。然后求s到t的最小割就是最小权值和。

```



### Dinic

```c++
template <typename F>//最大权闭合子图特殊情况
//首先建立源点s和汇点t，将源点s与所有权值为正的点相连，容量为权值；将所有权值为负的点与汇点t相连，
//容量为权值的绝对值；权值为0的点不做处理；同时将原来的边容量设置为无穷大
//结论：最大权闭合子图的权值等于所有正权点之和减去最小割。
struct Flow
{
    static constexpr F INF = numeric_limits<F>::max() / 2;
    struct Edge
    {
        int v;
        F cap;
        Edge(int v, F cap) : v(v), cap(cap) {}
    };
    int n;
    vector<Edge> e;
    vector<vector<int>> adj;
    vector<int> cur, h;
    Flow(){}
    Flow(int n) : n(n), adj(n) {}
    bool bfs(int s, int t)
    {
        h.assign(n, -1);
        queue<int> q;
        h[s] = 0;
        q.push(s);
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            for (int i : adj[u])
            {
                auto [v, c] = e[i];
                if (c > 0 && h[v] == -1)
                {
                    h[v] = h[u] + 1;
                    if (v == t)
                    {
                        return true;
                    }
                    q.push(v);
                }
            }
        }
        return false;
    }
    F dfs(int u, int t, F f)
    {
        if (u == t)
        {
            return f;
        }
        F r = f;
        for (int &i = cur[u]; i < adj[u].size(); i++)
        {
            int j = adj[u][i];
            auto [v, c] = e[j];
            if (c > 0 && h[v] == h[u] + 1)
            {
                F a = dfs(v, t, min(r, c));
                e[j].cap -= a;
                e[j ^ 1].cap += a;
                r -= a;
                if (r == 0)
                {
                    return f;
                }
            }
        }
        return f - r;
    }
    void addEdge(int u, int v, F cf = INF, F cb = 0)//单向边
    {
        adj[u].push_back(e.size()), e.emplace_back(v, cf);
        adj[v].push_back(e.size()), e.emplace_back(u, cb);
    }
    F maxFlow(int s, int t)
    {
        F ans = 0;
        while (bfs(s, t))
        {
            cur.assign(n, 0);
            ans += dfs(s, t, INF);
        }
        return ans;
    }
    // do max flow first
    vector<int> minCut()
    {
        vector<int> res(n);
        for (int i = 0; i < n; i++)
        {
            res[i] = h[i] != -1;
        }
        return res;
    }
};//Flow<int> mf(n+1);
```

### 预流推进网络流

```c++
namespace Dinic
{
	const int MAXN = 130005;
	const long long INF = 1e18;

	struct Node
	{
		int to, f;
	};

	vector<Node> pth[1500];
	vector<long long> Fl;
	int tot = -1;

	//	int H[MAXN], tot = 1;

	inline void Add_Edge(const int U, const int V, const long long f)
	{
		pth[U].push_back({V, ++tot}), Fl.push_back(f);
		pth[V].push_back({U, ++tot}), Fl.push_back(0);
	}

	int S, T;
	int Cur[MAXN], Dep[MAXN];

	inline bool BFS()
	{
		fill(Dep, Dep + MAXN, -1);
		queue<int> q;
		q.push(S), Dep[S] = 0;
		int u, to, f;
		while (!q.empty())
		{
			u = q.front(), q.pop(), Cur[u] = 0;
			for (auto i : pth[u])
			{
				to = i.to, f = Fl[i.f];
				if (Dep[to] == -1 && f)
				{
					Dep[to] = Dep[u] + 1, q.push(to);
					if (to == T)
						return 1;
				}
			}
		}
		return 0;
	}

	inline long long DFS(const int x, const long long MAXF)
	{
		if (x == T || MAXF == 0)
			return MAXF;
		long long F = 0;
		for (int i = Cur[x]; i < pth[x].size() && F < MAXF; ++i)
		{
			Cur[x] = i;
			if (Dep[pth[x][i].to] == Dep[x] + 1 && Fl[pth[x][i].f])
			{
				long long TmpF = DFS(pth[x][i].to, min(MAXF - F, Fl[pth[x][i].f]));
				if (!TmpF)
					Dep[pth[x][i].to] = -1;
				F += TmpF, Fl[pth[x][i].f] -= TmpF, Fl[pth[x][i].f ^ 1] += TmpF;
			}
		}
		return F;
	}

	inline long long dinic()
	{
		long long F = 0;
		while (BFS())
			F += DFS(S, INF);
		return F;
	}

	struct Node1
	{
		int u, v;
		long long c;

		inline bool operator<(const Node1 &a) const
		{
			return c > a.c;
		}
	};
	vector<Node1> V;

	inline void add_edge(const int u, const int v, const int w)
	{
		V.push_back({u, v, w});
	}

	inline long long GetAns()
	{
		long long Ans = 0;
		sort(V.begin(), V.end());
		for (int i = 1e9, j = 0; j < V.size(); i /= 20)
		{
			while (V[j].c >= i && j < V.size())
				Add_Edge(V[j].u, V[j].v, V[j].c), ++j;
			Ans += dinic();
		}
		return Ans;
	}
	void init(int n){
		Fl.clear();
	V.clear();
	tot = -1;
	for (int i = 0; i <= n; i++)
		pth[i].clear();
	}
}

// int main()
// {

//     ios::sync_with_stdio(0);
//     cin.tie(0), cout.tie(0);
//     using namespace Dinic;
//     int N, M, u, v, f;
//     cin >> N >> M >> S >> T;
//     for (int i = 1; i <= M; ++i)
//         cin >> u >> v >> f, Add_Edge(u, v, f);

//     cout << GetAns() << endl;

//     return 0;
// }
```



### 上下界网络流处理方法

```c++
void solveflow(){
    int n,m;cin>>n>>m;
    int S,T,s1=n+1,t1=n+2,s2=n+3,t2=n+4;//s1,t1:原图源汇点,s2,t2虚图源汇点
    vector<int> in(n + 6);Flow<int> flow(n + 6);
    for(int i=1;i<=m;i++){
        int u,v,l,r;cin>>u>>v>>l>>r;
        flow.addEdge(u,v,r-l);//建立边
        in[v]+=l;//入边加上下界
        in[u]-=l;//出边减去下界
    }int sum=0;
     for (int i = 1; i <= n + 2; i++)
    {
        if (in[i] > 0)
            flow.addEdge(s1, i, in[i]), sum += in[i];
        else
            flow.addEdge(i, t1, -in[i]);
    }
    flow.addEdge(T, S, INF);//汇向源建边
    if (flow.maxFlow(s1, t1) != sum)//无解
    {
        cout << -1 << endl;
           
        return;
    }
    int ans = prev(flow.e.end())->cap;//这个ans是可行流
    flow.adj[T].pop_back();
    flow.adj[S].pop_back();
    //ans+=flow.maxFlow(S, T);//有源汇上下界最大流
    //ans -= flow.maxFlow(T, S);//有源汇上下界最小流
    cout << ans << endl;         
}
```

### 费用流

```c++
const int MAXN = 2001;
const int MAXM = 8001;
const int INF=1e9;
//最小费用最大流
struct MCMF//如果求最大费用则费用取反
{
   
    int head[MAXN], cnt = 1;int maxflow, mincost;
    void init(int nu){//要跑多次先初始化
        fill(head, head + nu + 1, 0);
        maxflow = 0, mincost = 0;
        cnt = 1;
    }
    struct Edge
    {
        int to, w, c, next;
    } edges[MAXM * 2];
    inline void add(int from, int to, int w, int c)
    {
        edges[++cnt] = {to, w, c, head[from]};
        head[from] = cnt;
    }
    inline void addEdge(int from, int to, int w, int c)
    {
        add(from, to, w, c);
        add(to, from, 0, -c);
    }
    int s, t, dis[MAXN], cur[MAXN];
    bool inq[MAXN], vis[MAXN];
    queue<int> Q;
    bool SPFA()
    {
        while (!Q.empty())
            Q.pop();
        copy(head, head + MAXN, cur);
        fill(dis, dis + MAXN, INF);
        dis[s] = 0;
        Q.push(s);
        while (!Q.empty())
        {
            int p = Q.front();
            Q.pop();
            inq[p] = 0;
            for (int e = head[p]; e != 0; e = edges[e].next)
            {
                int to = edges[e].to, vol = edges[e].w;
                if (vol > 0 && dis[to] > dis[p] + edges[e].c)
                {
                    dis[to] = dis[p] + edges[e].c;
                    if (!inq[to])
                    {
                        Q.push(to);
                        inq[to] = 1;
                    }
                }
            }
        }
        return dis[t] != INF;
    }
    int dfs(int p, int flow)
    {
        if (p == t)
            return flow;
        vis[p] = 1;
        int rmn = flow;
        for (int eg = cur[p]; eg && rmn; eg = edges[eg].next)
        {
            cur[p] = eg;
            int to = edges[eg].to, vol = edges[eg].w;
            if (vol > 0 && !vis[to] && dis[to] == dis[p] + edges[eg].c)
            {
                int c = dfs(to, min(vol, rmn));
                rmn -= c;
                edges[eg].w -= c;
                edges[eg ^ 1].w += c;
            }
        }
        vis[p] = 0;
        return flow - rmn;
    }
    
    inline void run(int ss, int tt)
    {
        s = ss, t = tt;
        while (SPFA())
        {
            int flow = dfs(s,INF);
            maxflow += flow;
            mincost += dis[t] * flow;
        }
    }
}mcmf; // namespace MCMF
```

### 最大费用可行流

```c++
const int MAXN = 850;
const int MAXM = 500005;
const int INF=1e9;
struct MCMF // 最大费用可行流
{

    int head[MAXN], cnt = 1;
    int maxflow, mincost;
    void init(int nu)
    { // 要跑多次先初始化
        fill(head, head + nu + 1, 0);
        maxflow = 0, mincost = 0;
        cnt = 1;
    }
    struct Edge
    {
        int to, w, c, next;
    } edges[MAXM * 2];
    inline void add(int from, int to, int w, int c)
    {
        edges[++cnt] = {to, w, c, head[from]};
        head[from] = cnt;
    }
    inline void addEdge(int from, int to, int w, int c)
    {
        add(from, to, w, c);
        add(to, from, 0, -c);
    }
    int s, t, dis[MAXN], cur[MAXN];
    bool inq[MAXN], vis[MAXN];
    int q[4000001], l = 1, r = 0;
    bool SPFA()
    {
        l = 1, r = 0;
        copy(head, head + MAXN, cur);
        fill(dis, dis + MAXN, INF);
        dis[s] = 0;
        q[++r] = s;
        while (r >= l)
        {
            int p = q[l++];
            inq[p] = 0;
            for (int e = head[p]; e != 0; e = edges[e].next)
            {
                int to = edges[e].to, vol = edges[e].w;
                if (vol > 0 && dis[to] > dis[p] + edges[e].c)
                {
                    dis[to] = dis[p] + edges[e].c;
                    if (!inq[to])
                    {
                        q[++r] = to;
                        inq[to] = 1;
                    }
                }
            }
        }
        if (dis[t] == INF || dis[t] >= 0)
            return 0;
        return 1;
    }
    int dfs(int p, int flow)
    {
        if (p == t)
            return flow;
        vis[p] = 1;
        int rmn = flow;
        for (int eg = cur[p]; eg && rmn; eg = edges[eg].next)
        {
            cur[p] = eg;
            int to = edges[eg].to, vol = edges[eg].w;
            if (vol > 0 && !vis[to] && dis[to] == dis[p] + edges[eg].c)
            {
                int c = dfs(to, min(vol, rmn));
                rmn -= c;
                edges[eg].w -= c;
                edges[eg ^ 1].w += c;
            }
        }
        vis[p] = 0;
        return flow - rmn;
    }

    inline void run(int ss, int tt)
    {
        s = ss, t = tt;
        while (SPFA())
        {
            int flow = dfs(s, INF);
            maxflow += flow;
            mincost += dis[t] * flow;
        }
    }
} mcmf; // namespace MCMF
```

### 上下界最小费用可行/最大流

```c++
inline void Solve()
    {

        cin >> n;
        int s1 = n + 1, t1 = n + 2, s2 = n + 3, t2 = n + 4; // s1,t1:原图源汇点,s2,t2虚图源汇点
        vector<int> in(n + 5);
        for (int i = 1; i <= n; i++)
        {
            int t;
            cin >> t;
            if (i != 1)
            {
                Add_Edge(i, 1, INF, 0);
            }
            for (int j = 1; j <= t; j++)
            {
                int to, co;
                cin >> to >> co;
                MinC += co;//最小费用要加上下界
                Add_Edge(i, to, INF - 1, co);
                in[to] += 1;
                in[i] -= 1;
            }
        }
        S = s2, T = t2;
        for (int i = 1; i <= n; i++)
        {
            if (in[i] > 0)
            {
                Add_Edge(s2, i, in[i], 0);
            }
            else
            {
                Add_Edge(i, t2, -in[i], 0);
            }
        }
        Add_Edge(T, S, INF, 0);
        long long MaxF = Simplex();

        //cout << MinC << endl;最小费用可行流,最大流删除最后两条边从原图源汇点出发再跑一次

    }
```

### 费用流加强版

```c++
const int MAXN = 100005;
const long long INF = 1e12;
#define int long long
using namespace std;

namespace MCMF
{
    struct Edge
    {
        int u, v, nxt;
        long long f, w;
    } E[500005];

    int H[MAXN], tot = 1;

    inline void Add_Edge(const int u, const int v, const long long f, const long long w)
    {
        E[++tot] = {u, v, H[u], f, +w}, H[u] = tot;
        E[++tot] = {v, u, H[v], 0, -w}, H[v] = tot;
    }

    long long Pre[MAXN];
    int fa[MAXN], fe[MAXN], Cir[MAXN];
    int Tag[MAXN], Now = 1;
    int S = 11451, T = 19198;

    inline void Init_ZCT(const int x, const int e)
    { // Make a Random Zhicheng Tree
        fe[x] = e, fa[x] = E[fe[x]].u, Tag[x] = 1;
        for (int i = H[x]; i; i = E[i].nxt)
            if (E[i].f && !Tag[E[i].v])
                Init_ZCT(E[i].v, i);
    }

    inline long long Sum(const int x)
    {
        if (Tag[x] == Now)
            return Pre[x];
        Tag[x] = Now, Pre[x] = Sum(fa[x]) + E[fe[x]].w;
        return Pre[x];
    }

    inline long long Push_Flow(const int x)
    {
        // Find LCA (Top of Circle)
        int rt = E[x].u, lca = E[x].v, Cnt = 0;
        ++Now;
        while (rt)
            Tag[rt] = Now, rt = fa[rt];
        while (Tag[lca] != Now)
            Tag[lca] = Now, lca = fa[lca];

        // Find Circle
        long long F = E[x].f;
        int Del = 0, P = 2;
        for (int u = E[x].u; u != lca; u = fa[u])
        {
            Cir[++Cnt] = fe[u];
            if (E[fe[u]].f < F)
                F = E[fe[u]].f, P = 0, Del = u;
        }

        for (int u = E[x].v; u != lca; u = fa[u])
        {
            Cir[++Cnt] = fe[u] ^ 1;
            if (E[fe[u] ^ 1].f < F)
                F = E[fe[u] ^ 1].f, P = 1, Del = u;
        }
        Cir[++Cnt] = x;

        // Push Flow
        long long Cost = 0;
        for (int i = 1; i <= Cnt; ++i)
            Cost += E[Cir[i]].w * F, E[Cir[i]].f -= F, E[Cir[i] ^ 1].f += F;

        if (P == 2)
            return Cost; // MinFlow on Edge You Add

        int u = E[x].u, v = E[x].v;
        if (P == 1)
            swap(u, v);
        int Lste = x ^ P, Lstu = v, Tmp;

        while (Lstu != Del)
        {
            Lste ^= 1, --Tag[u];
            swap(fe[u], Lste);
            Tmp = fa[u], fa[u] = Lstu, Lstu = u, u = Tmp;
        }
        return Cost;
    }

    long long MinC = 0;

    inline long long Simplex()
    {
        Add_Edge(T, S, INF, -INF);
        Init_ZCT(T, 0);
        Tag[T] = ++Now, fa[T] = 0;

        bool Run = 1;
        while (Run)
        {
            Run = 0;
            for (int i = 2; i <= tot; ++i)
                if (E[i].f && E[i].w + Sum(E[i].u) - Sum(E[i].v) < 0)
                    MinC += Push_Flow(i), Run = 1;
        }

        MinC += E[tot].f * INF;
        return E[tot].f;
    }
}

// namespace Value
// {
//     using namespace MCMF;
//     int N, M, u, v, f, w;

//     inline void Solve()
//     {

//         ios::sync_with_stdio(0);
//         cin.tie(0), cout.tie(0);

//         cin >> N >> M >> S >> T;

//         for (int i = 1; i <= M; ++i)
//         {
//             cin >> u >> v >> f>>w;
//             Add_Edge(u, v, f, 0);
//         }

//         long long MaxF = Simplex();

//         cout << MaxF << endl;
//     }
// }
```

### KM(二分图最大匹配)

``` cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
// #define int long long
const int INF = 0x7fffffff;
ll b[404], c[404], r, v[404];

pair<ll, ll> p[404];
const ll Maxn = 505;
const ll inf = 1e18;
ll n, m, mp[Maxn][Maxn], matched[Maxn];
ll slack[Maxn], pre[Maxn], ex[Maxn], ey[Maxn]; // ex,ey顶标
bool visx[Maxn], visy[Maxn];
void match(ll u)
{
    ll x, y = 0, yy = 0, delta;
    memset(pre, 0, sizeof(pre));
    for (ll i = 1; i <= n; i++)
        slack[i] = inf;
    matched[y] = u;
    while (1)
    {
        x = matched[y];
        delta = inf;
        visy[y] = 1;
        for (ll i = 1; i <= n; i++)
        {
            if (visy[i])
                continue;
            if (slack[i] > ex[x] + ey[i] - mp[x][i])
            {
                slack[i] = ex[x] + ey[i] - mp[x][i];
                pre[i] = y;
            }
            if (slack[i] < delta)
            {
                delta = slack[i];
                yy = i;
            }
        }
        for (ll i = 0; i <= n; i++)
        {
            if (visy[i])
                ex[matched[i]] -= delta, ey[i] += delta;
            else
                slack[i] -= delta;
        }
        y = yy;
        if (matched[y] == -1)
            break;
    }
    while (y)
    {
        matched[y] = matched[pre[y]];
        y = pre[y];
    }
}
ll KM()
{
    memset(matched, -1, sizeof(matched));
    memset(ex, 0, sizeof(ex));
    memset(ey, 0, sizeof(ey));
    for (ll i = 1; i <= n; i++)
    {
        memset(visy, 0, sizeof(visy));
        match(i);
    }
    ll res = 0;
    for (ll i = 1; i <= n; i++)
        if (matched[i] != -1)
            res += mp[matched[i]][i];
    return res;
}
void solve()
{
    cin >> n;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            mp[i][j] = -inf;
    for (int i = 1; i <= n; i++)
        cin >> p[i].first; // 战力
    for (int i = 1; i <= n; i++)
    {
        cin >> p[i].second; // 获得
        // cout << p[i].second << endl;
    }
    // mcmf.init(n * 2 + 5);
    int ss = n * 2 + 1, tt = n * 2 + 2;
    for (int i = 1; i <= n; i++)
        cin >> b[i];
    for (int i = 1; i <= n; i++)
        cin >> c[i];
    for (int i = 1; i <= n; i++)
    {

        for (int j = 1; j <= n; j++)
        {
            ll nu = b[i] + c[j], val = 0;
            for (int k = 1; k <= n; k++)
            {
                if (nu > p[k].first)
                {
                    val += p[k].second;
                }
            }
            // cout << val << endl;
            mp[i][j] = val;
        }
    }

    cout << KM() << endl;
    // cout << cnt << endl;
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
        solve();
}

```



### 2-sat

```cpp
struct SCC {
    vector<vector<int>> g, scc;//scc的编号满足反拓扑序,倒过来从cnt到1更新即可.
    vector<int> stk, l, id, ins, low, sz;
    int n;
    int cnt, top, idx;
    vector<vector<int>> ng;
    SCC() {}
    SCC(int _n) :n(_n), cnt(0), top(0), idx(0) {
        g.resize(n + 1);
        scc.resize(n + 1);
        stk.resize(n + 1);
        l.resize(n + 1);
        id.resize(n + 1);
        ins.resize(n + 1);
        low.resize(n + 1);
        sz.resize(n + 1);
    }
    void AddEdge(int u, int v) {
        g[u].push_back(v);
    }
    int size() { return cnt; }
    vector<int> operator[](const int& k)const { return scc[k]; }
    vector<int>& operator[](const int& k) { return scc[k]; }
    int belong(int x) { return id[x]; }
    void run() {
        auto dfs = [&](auto&& dfs, int u)->void {
            low[u] = l[u] = ++idx;
            stk[++top] = u;
            ins[u] = 1;
            for (auto v : g[u]) {
                if (!l[v]) dfs(dfs, v);
                if (ins[v]) low[u] = min(low[u], low[v]);
            }
            if (low[u] == l[u]) {
                cnt += 1;
                while (top) {
                    int v = stk[top--];
                    scc[cnt].push_back(v);
                    ins[v] = 0;
                    id[v] = cnt;
                    sz[cnt] += 1;
                    if (u == v) break;
                }
            }
            };
        for (int i = 1;i <= n;i++) {
            if (!l[i]) dfs(dfs, i);
        }
    }
    void build() {
        ng.resize(cnt + 1);
        unordered_set<int> st;
        for (int u = 1;u <= n;u++) {
            for (auto v : g[u]) {
                if (id[u] == id[v]) continue;
                int t = id[u] * 1919810 + id[v];
                if (st.count(t)) continue;
                ng[id[u]].push_back(id[v]);
                st.insert(t);
            }
        }
    }
};

struct Two_SAT {
    int n;SCC scc;
    Two_SAT(int _n) :n(_n) {
        scc = SCC(n * 2);
    }
    void Add(int x, int y) {
        scc.AddEdge(x, y);
    }
    vector<int> operator[](const int& k)const { return scc[k]; }
    vector<int>& operator[](const int& k) { return scc[k]; }
    int size() {
        return scc.size();
    }
    void run() {
        scc.run();
    }
    int belong(int x) {
        return scc.belong(x);
    }

    int inv(int x) {
        if (x <= n) return x + n;
        return x - n;
    }
    void implies(int u, int v) {//u蕴含v，即u为真则v为真
        scc.AddEdge(u, v);
        scc.AddEdge(inv(v), inv(u));
    }
    void either(int u, int v) {//u和v中至少一个为真
        scc.AddEdge(inv(u), v);
        scc.AddEdge(inv(v), u);
    }
    void equal(int u, int v) {//u=v
        implies(u, v);
        implies(inv(u), inv(v));
    }
    void unequal(int u, int v) {//u!=v
        implies(u, inv(v));
        implies(inv(u), v);
    }
    void set(int u) {//u=1
        scc.AddEdge(inv(u), u);
    }

    int solve() {
        run();
        for (int i = 1;i <= n;i++) {
            if (belong(i) == belong(i + n)) return 0;
        }
        return 1;
    }
    int get(int x) {
        return (belong(x) < belong(x + n));
    }
};
```



### 最小生成树

#### kruscal

```c++
struct DSU {
    vector<int> p, siz;
    DSU(int n) :p(n), siz(n, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) {
        return p[x] == x ? x : p[x] = find(p[x]);
    }
    bool same(int x, int y) {
        return find(x) == find(y);
    }
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return 0;
        if (siz[x] < siz[y]) {
            swap(x, y);
        }
        siz[x] += siz[y];
        p[y] = x;
        return 1;
    }
    int size(int x) {
        return siz[find(x)];
    }
};

void Solve(int TIME) {

    int n, m;cin >> n >> m;
    vc<array<int, 3>> g;
    DSU d(n + 1);
    for (int i = 1;i <= m;i++) {
        int u, v, w;cin >> u >> v >> w;
        g.push_back({ w,u,v });
    }

    sort(g.begin(), g.end());
    int res = 0;
    for (auto [w, u, v] : g) {
        if (d.find(u) != d.find(v)) d.merge(u, v), res += w;
    }
    if (d.size(1) != n) return cout << "NO" << endl, void();
    cout << res << endl;

}
```



### 克鲁斯卡尔重构树

```cpp
int find(int x){return x==fa[x]?x:fa[x]=find(fa[x]);}
void Kruskal()
{
	sort(E+1, E+m+1, cmp);
	for (int i=1; i<=n; i++) fa[i]=i; int now=n;
	for (int i=1; i<=m; i++)
	{
    	int x=E[i].x, y=E[i].y, w=E[i].w;
    	int fx=find(x), fy=find(y);
    	if (fx^fy)
    	{
        	val[++now]=w; fa[fx]=fa[fy]=fa[now]=now; 
        	add(now, fx, 1); add(now, fy, 1);
    	}
	}
}
//1、它是一个二叉堆。

//2、若边权升序，则它是一个大根堆

//3、任意两点路径边权最大值为重构树上LCA的点权。
```

#### 01异或最小生成树

```cpp
const int inf = 0x3f3f3f3f;//边权为a^b
const int maxn = 1e7 + 7, maxm = 3e6 + 7;
struct Xortrie
{
    int node;             /// 节点编号
    int son[maxn][2];     /// 01字典树
    int L[maxn], R[maxn]; /// 该节点控制的左区间和右区间
    void Xortrie_init()
    {
        for(int i=0;i<=node;i++){
            son[i][1]=son[i][0]=0;
        }
        edge.clear();
        node = 0; /// 初始化
    }
    /// 插入这个数 需要控制左右区间
    void add(ll x, int id)
    {
        int rt = 0;
        for (int i = 31; i >= 0; i--)
        {
            /// 逐位分离
            int tmp = (x >> i & 1ll) ? 1 : 0; /// 判断当前位
            int &tt = son[rt][tmp];           /// 当前节点是否存在
            if (!tt)
                tt = ++node; /// 如果被卡内存，可以改用动态开点
            rt = tt;
            /// 如何记录一个节点控制的左右区间
            if (!L[rt])
                L[rt] = id; /// 最早的是左端点
            R[rt] = id;     /// 一直往右拓展
        }
    }
    /// 询问rt从pos位开始和x异或得到的min和下标
    pair<int, int> qaskpos(int rt, int pos, ll x)
    {
        ll res = 0, id = 0;
        for (int i = pos; i >= 0; i--)
        {
            int tmp = (x >> i & 1ll) ? 1 : 0; /// 判断当前位
            if (son[rt][tmp])
                rt = son[rt][tmp];
            else
            {
                res += (1ll << i);
                rt = son[rt][!tmp];
            }
        }
        id = L[rt];
        return {res, id};
    }
    vector<pair<int, int>> edge;
    /// 分治操作 查询某子树
    ll Divide(int rt, int pos)
    {

        if (son[rt][0] && son[rt][1])
        {
            /// 左右子树均存在
            int x = son[rt][0], y = son[rt][1];
            ll minn = INF;
            /// 启发式合并:根据子树大小合并，小优化
            int u, v;
            if (R[x] - L[x] + 1 <= R[y] - L[y] + 1)
            {                                      /// 如果左子树小
                for (int k = L[x]; k <= R[x]; k++) /// 枚举左子树
                {
                    auto [nu, id] = qaskpos(y, pos - 1, a[k]);
                    if (minn > nu + (1ll << pos))
                    {
                        minn = nu + (1ll << pos);
                        u = k, v = id;
                    }
                }
            }
            else
            {
                for (int k = L[y]; k <= R[y]; k++) /// 枚举右子树
                {
                    auto [nu, id] = qaskpos(x, pos - 1, a[k]);
                    if (minn > nu + (1ll << pos))
                    {
                        minn = nu + (1ll << pos);
                        u = k, v = id;
                    }
                }
            }
            edge.push_back({u, v});
            return minn + Divide(x, pos - 1) + Divide(y, pos - 1); /// 左右子树合并的最小异或值+左子树的最小异或值+右子树的最小异或值
        }
        else if (son[rt][0])
        {
            /// 只有左子树 递归
            return Divide(son[rt][0], pos - 1);
        }
        else if (son[rt][1])
        {
            /// 只有右子树 递归
            return Divide(son[rt][1], pos - 1);
        }
        /// 叶子节点
        return 0ll;
    }
    void debug(int rt){
        cout<<rt<<" "<<L[rt]<<" "<<R[rt]<<endl;
        if(son[rt][0]) debug(son[rt][0]);
        if(son[rt][1]) debug(son[rt][1]);
    }
    int get(){
        return Divide(0,31);
    }
} az;

// signed main()
// {
//     az.Xortrie_init();
//     cin >> n;
//     for (int i = 1; i <= n; i++)
//         cin >> a[i];
//     sort(a + 1, a + 1 + n);
//     for (int i = 1; i <= n; i++)
//         az.add(a[i], i);

//      ll q=az.Divide(0,31);
//     return 0;
// }
```



#### prim

```c++
int n, m;
int g[N][N];
bool st[N];
int dist[N];//该点到已经存入集合的点的距离

//最小生成树prim  O(n²)  稠密图
int prim() {
    fill(begin(dist), end(dist), inf);
    int res = 0;
    for (int i = 0; i < n; i++) {
        int t = -1;
        for (int j = 1; j <= n; j++) {
            if (!st[j] && t == -1 || dist[t] > dist[j])
                t = j;
        }
        if (i > 0 && dist[t] == inf) return inf;
        if (i > 0) res += dist[t];
        for (int j = 1; j <= n; j++) dist[j] = min(dist[j], g[t][j]);
        st[t] = true;
    }
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> m;
    fill(&g[0][0], &g[0][0] + N * N, inf);
    while (m--) {
        int a, b, w;
        cin >> a >> b >> w;
        g[a][b] = w;
        g[a][b] = g[b][a] = min(g[a][b], w);
    }
    int t = prim();
    if (t == inf) cout << "Impossible" << endl;
    else cout << t << endl;
    return 0;
}
```

### 最短路

#### dijkstra

```c++
vi dis(n + 1);
auto dijkstra = [&](int s, vi& dis) {
    for (int i = 0;i <= n;i++) dis[i] = inf;
    vi vis(n + 1);
    dis[s] = 0;
    priority_queue<array<int, 2>, vector<array<int, 2>>, greater<array<int, 2>>> q;q.push({ dis[s],s });
    while (q.size()) {
        auto [d, u] = q.top();q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (auto [v, w] : g[u]) {
            if (dis[v] > dis[u] + w) {
                dis[v] = dis[u] + w;
                q.push({ dis[v],v });
            }
        }
    }
    };
dijkstra(s, dis);
```



### Tarjan求强连通分量

```c++
    vector<int>co(n + 1), dfn(n + 1), low(n + 1),stk,siz(n+1);//强联通分量染色，dfs序,缩点处,栈，强联通分量大小
    int cnt=0,tot=0;//点数，缩点后的点数
    auto tarjan = [&](int pos, int fa, auto tarjan) -> void
    {
        dfn[pos] = low[pos] = ++cnt;
        stk.push_back(pos);
        for(auto &[to,len]:pth[pos]){
            if(to==fa)
                continue;
            if (!dfn[to])
            {
                tarjan(to, pos, tarjan);
                low[pos] = min(low[pos], low[to]);
            }
            else if (!co[to])
            {
                low[pos] = min(low[pos], dfn[to]);
            }
        }
        if (dfn[pos] == low[pos])
        {
            ++tot;
            while (stk.back() != pos)
            {
                co[stk.back()] = tot;
                siz[tot]++;
                stk.pop_back();
            }
            co[stk.back()] = tot;
            siz[tot]++;
            stk.pop_back();
        }
    };
```



### 2-SAT

```c++
n = read(), m = read();//给定条件x[a]为va或x[b]为v[b]
for (int i = 0; i < m; ++i) {
    // 笔者习惯对 x 点标号为 x，-x 标号为 x+n，当然也可以反过来。
    int a = read(), va = read(), b = read(), vb = read();
    if (va && vb) { // a, b 都真，-a -> b, -b -> a
        g[a + n].push_back(b);
        g[b + n].push_back(a);
    } else if (!va && vb) { // a 假 b 真，a -> b, -b -> -a
        g[a].push_back(b);
        g[b + n].push_back(a + n);
    } else if (va && !vb) { // a 真 b 假，-a -> -b, b -> a
        g[a + n].push_back(b + n);
        g[b].push_back(a);
    } else if (!va && !vb) { // a, b 都假，a -> -b, b -> -a
        g[a].push_back(b + n);
        g[b].push_back(a + n);
    }
}//之后运行tarjan,-a和a在同一个强连通分量里则无解
```



### rmq求树上距离和lca

```c++
struct RMQ{//求序列中区间最大？最小值的下标
    int n;
    vector<int> lg,nu;
    vector<array<int, 25>> dp;//固定内部数组降低常数
    int get(const int& a,const int& b){
        return nu[a] > nu[b] ? a : b;//自定义比较函数，返回在原数组中较大的下标
    }
    RMQ(vector<int> &v){//v的有效下标从1开始，v.size()-1结束
        n = v.size()-1;
        nu = v;
        lg.resize(n+1);
        dp.resize(n+1);
        lg[1] = 0;
        for (int i = 2; i <= n;i++){
            lg[i] = lg[i >> 1] + 1;
        }
        for (int i = 1; i <= n;i++){
            dp[i][0] = i;
        }
        for (int j = 1; j <= lg[n]; j++)
         {
            for (int i = 1; i <= n - (1 << j) + 1; i++)
            {
                dp[i][j] = get(dp[i][j - 1], dp[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
    int getidex(const int &l, const int &r)//返回最值下标
    {
        int len = lg[r - l + 1];
        return get(dp[l][len], dp[r - (1 << len) + 1][len]);
    }
    int getnum(const int &l, const int &r) // 返回最值
    {
        return nu[getidex(l, r)];
    }
};
struct GRAPH{
    vector<vector<pair<int,int>>> pth;
    vector<int> dep,first,nu,ref;
    int cnt = 0;
    GRAPH(int n){
        pth.resize(n + 1);
        dep.resize(n + 1);
        first.resize(n + 1);
        nu.resize(n*2 + 1);
        ref.resize(n * 2 + 1);
        cnt = 0;//构造好了以后直接加边
    }
    void add_edge(int u,int v,int w=1){//加双向边
        pth[u].push_back({v, w});
        pth[v].push_back({u, w});
    }
    void dfs(int pos,int fa)
    {
        first[pos] = ++cnt; // 第一次碰到这个点的dfs序
        nu[cnt] = dep[pos]; // 这个dfs序的深度
        ref[cnt] = pos;
        for (auto [to, len] : pth[pos])
        {
            if (to == fa)
                continue;
            dep[to] = dep[pos] + len; // 当前点深度
            dfs(to,pos);
            nu[++cnt] = dep[pos];
            ref[cnt] = pos;
        }
    };
    RMQ rmq;
    void prepare(int st){//st是树的根节点
        dfs(st, 0);
        rmq = RMQ(nu);
    }
    int query_lca(int x, int y){ // 询问两点间的最近公共祖先
        int l, r;
        l = first[x], r = first[y];
        if (l > r)
            swap(l, r);
        int t = rmq.getidex(l, r);
        int lca = ref[t];
        return lca;
    };
    int query_dis(int x, int y)
    { // 询问两点间的距离
        int len = dep[x] + dep[y] - 2 * dep[query_lca(x,y)];
        return len;
    };
};
```

### 线段树辅助建边

```c++
int po1[200005],n,dif,po2[200005],dis[800005];
struct link_star{
    struct link{
        int fore,to,len;
    };
    int cnt;
    vector<int> head;
    vector<link> node;
    void init(int n,int m){
        cnt = 0;
        head.resize(n + 1);
        node.resize((m + 1) << 1);
    }
    link_star(int n,int m) : cnt(0) { init(n,m); }
    link_star(){}
    void build(int a,int b,int len){
        node[++cnt].fore = head[a];
        node[cnt].to = b;
        node[cnt].len = len;
        head[a] = cnt;
    }
    void add_edge(int a,int b,int len){
        build(a,b,len);
        build(b, a, len);
    }
 
}edge;
 
void build1(int p,int l,int r){//入边树
    if(l==r){
        edge.add_edge(p, p + dif, 0);
        po1[l] = p;
        return;
    }
    edge.build(p, p<<1, 0);
    edge.build(p, p<<1|1, 0);
    int mid = l + r >> 1;
    build1(p << 1, l, mid);
    build1(p << 1 | 1, mid + 1, r);
}
void build2(int p, int l, int r)
{ // 出边树
    if (l == r)
    {
        po2[l] = p;
        return;
    }
    edge.build((p << 1) + dif, p + dif, 0);
    edge.build((p << 1 | 1) + dif, p + dif, 0);
    int mid = l + r >> 1;
    build2(p << 1, l, mid);
    build2(p << 1 | 1, mid + 1, r);
}
void add1(int p,int l,int r,int ql,int qr,int po,int w){
 
    if(l>=ql&&r<=qr){
        edge.build(po2[po] + dif, p, w);
        return;
    }
    int mid = l + r >> 1;
    if(ql<=mid)
        add1(p << 1, l, mid, ql, qr, po, w);
    if(qr>mid)
        add1(p << 1|1, mid+1, r, ql, qr, po, w);
}
void add2(int p, int l, int r, int ql, int qr, int po, int w)
{
    if (l >= ql && r <= qr)
    {
        edge.build(p + dif,po1[po] , w);
        return;
    }
    int mid = l + r >> 1;
    if (ql <= mid)
        add2(p << 1, l, mid, ql, qr, po, w);
    if (qr > mid)
        add2(p << 1 | 1, mid + 1, r, ql, qr, po, w);
}
// if (t == 1)
// {
//     int u, v, w;
//     cin >> u >> v >> w;
//     edge.build(po2[u] + dif, po1[v], w);
// }
// else if (t == 2)v向[l,r]
// {
//     int u, l, r, w;
//     cin >> u >> l >> r >> w;
//     add1(1, 1, n, l, r, u, w);
// }
// else
// {
//     int u, l, r, w;
//     cin >> u >> l >> r >> w;
//     add2(1, 1, n, l, r, u, w);
// }
```



## 计算几何

### 简易版计算集合(int)

``` cpp
#define debugP(x) cerr<<#x<<":("<<x[0]<<','<<x[1]<<")"<<endl
using ld = long double;
using P = array<int, 2>;
using LI = array<P, 2>;
using PD = array<ld, 2>;
using LD = array<PD, 2>;
using CI = vector<P>;
using CD = vector<PD>;
const ld eps = 1e-9;
 
int prev(int i, int n) {
    return i == 0 ? n - 1 : i - 1;
}
int next(int i, int n) {
    return i == n - 1 ? 0 : i + 1;
}
 
int sgn(ld x) {
    return x > eps ? 1 : (x < -eps ? -1 : 0);
}
 
P uv(P u, P v) {
    return { v[0] - u[0],v[1] - u[1] };
}
P add(P u, P v) {
    return { u[0] + v[0],u[1] + v[1] };
}
int dis2(P u, P v) {
    return (u[0] - v[0]) * (u[0] - v[0]) + (u[1] - v[1]) * (u[1] - v[1]);
}

int cross(P u, P v) {
    return u[0] * v[1] - u[1] * v[0];
}
 
int dot(P u, P v) {
    return u[0] * v[0] + u[1] * v[1];
}
 
int loca(P u, P v, P w) {
    return sgn(cross(uv(u, v), uv(u, w)));
}
int locb(P u, P v, P w) {
    return sgn(dot(uv(u, v), uv(u, w)));
}
 
struct argcmp {//极角排序
    inline static int DS[4] = { 1,2,4,3 };
    bool operator()(const P& a, const P& b)const {
        const auto quad = [&](const P& u) {
            return DS[(sgn(u[1]) < 0) * 2 + (sgn(u[0]) < 0)];
            };
        int c = quad(a), d = quad(b);
        if (c != d) return c < d;
        return sgn(cross(a, b)) > 0;
    }
};
 
bool point_on_segment(P a, LI line) {
    return sgn(cross(uv(a, line[0]), uv(a, line[1]))) == 0
        && sgn(dot(uv(a, line[0]), uv(a, line[1]))) <= 0;
}
 
bool inter_judge_segment(LI a, LI b) {
    //一个点的端点在另一个线段
    if (point_on_segment(b[0], a)
        || point_on_segment(b[1], a)
        || point_on_segment(a[0], b)
        || point_on_segment(a[1], b))
        return 1;
    //跨立试验
    return (loca(a[0], b[0], a[1]) * loca(a[0], b[1], a[1]) < 0
        && loca(b[0], a[0], b[1]) * loca(b[0], a[1], b[1]) < 0
        );
}
 
bool inter_judge(LI a, LI b) {
    return sgn(cross(uv(a[0], a[1]), uv(a[0], b[0]))
        - cross(uv(a[0], a[1]), uv(a[0], b[1]))) != 0;
}
 
vector<P> convex_hull(vector<P> a) {//Andrew求凸包(扫描线)
    if (a.size() <= 2) return a;
    sort(a.begin(), a.end());
    vector<P> ret;
    for (int i = 0;i < a.size();i++) {
        while (ret.size() >= 2 && loca(ret[ret.size() - 2], ret[ret.size() - 1], a[i]) <= 0)
            ret.pop_back();
        ret.push_back(a[i]);
    }
    int fixed = ret.size();
    for (int i = (int)a.size() - 2;i >= 0;i--) {
        while (ret.size() > fixed && loca(ret[ret.size() - 2], ret[ret.size() - 1], a[i]) <= 0)
            ret.pop_back();
        ret.push_back(a[i]);
    }
    ret.pop_back();
    return ret;
}
array<int, 2> in_convex(P p, const CI& a) {//{no | strictly yes | yes,where}
    int n = a.size();
    if (n == 1) {
        return { sgn(p[0] - a[0][0]) == 0 && sgn(p[1] - a[0][0]) == 0,0 };
    }
    if (n == 2) {
        return { point_on_segment(p, { a[0],a[1] }),0 };
    }
    int l = 1, r = n - 2;
    while (l <= r) {
        int mid = l + r >> 1;
        int u = loca(a[0], a[mid], p);
        int v = loca(a[0], a[mid + 1], p);
        if (u >= 0 && v <= 0) {
            if (loca(a[mid], a[mid + 1], p) >= 0) {
                {//在凸包的边上
                    if (loca(a[mid], a[mid + 1], p) == 0) return { 2,mid };
                    if (mid == 1 && loca(a[mid], a[0], p) == 0) return { 2,0 };
                    if (mid + 1 == n - 1 && loca(a[mid + 1], a[0], p) == 0) return { 2,n - 1 };
                }
                return { 1,mid };
            }
            return { 0,0 };
        }
        if (u < 0) r = mid - 1;
        else l = mid + 1;
    }
    return { 0,0 };
}

int diam2(const CI& a) {//直径平方
    int r = 0;
    int n = a.size();
    if (n <= 2) {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                r = max(r, dis2(a[i], a[j]));
        return r;
    }
    for (int i = 0, j = 1; i < n; i++) {
        while (
            cross(uv(a[i], a[next(i, n)]), uv(a[i], a[j]))
            - cross(uv(a[i], a[next(i, n)]), uv(a[i], a[next(j, n)])) <= 0
            ) j = next(j, n);
        r = max({ r, dis2(a[i], a[j]), dis2(a[next(i, n)], a[j]) });
    }
    return r;
}
int area2(const vector<P>& a) {
    int ret = 0;
    for (int i = 0;i < a.size();i++) {
        int j = (i + 1) % a.size();
        ret += cross(a[i], a[j]);
    }
    return abs(ret);
}
 
P find_max(const CI& a, auto cmp) {//极点
    int l = 1, r = (int)a.size() - 2;
    if (cmp(a.back(), a[0])) {
        while (l <= r) {
            int mid = l + r >> 1;
            if (cmp(a[l - 1], a[mid]) && cmp(a[mid - 1], a[mid])) l = mid + 1;
            else r = mid - 1;
        }
        return a[r];
    }
    else {
        while (l <= r) {
            int mid = l + r >> 1;
            if (cmp(a[r + 1], a[mid]) && cmp(a[mid + 1], a[mid])) r = mid - 1;
            else l = mid + 1;
        }
        return a[l];
    }
}
 
//The order of the answer is counterclockwise of the convex hull.
array<P, 2> tangent(const CI& a, P u) {//过u的切线
    //如果点在凸包的点和边需要特判
    return { find_max(a,[&](auto x,auto y) {return loca(u,x,y) > 0;}),
            find_max(a,[&](auto x,auto y) {return loca(u,y,x) > 0;}) };
}
 
array<P, 2> tangent_vec(const CI& a, P u) {//与u平行的切线
    //如果点在凸包的点和边需要特判
    return { find_max(a,[&](auto x,auto y) {return sgn(cross(u,uv(x,y))) > 0;}),
            find_max(a,[&](auto x,auto y) {return sgn(cross(u,uv(y,x))) > 0;}) };
}
 
 
CI minkovski(vector<CI> C) {//结果是非严格凸包，即存在三点共线(可以通过再跑一遍凸包算法变成严格凸包)
    auto run = [&](array<CI, 2> a) {
        for (int i = 0;i < 2;i++) a[i].push_back(a[i].front());
        int i[2] = { 0,0 }, j[2] = { 0,0 }, len[2] = { (int)a[0].size() - 1,(int)a[1].size() - 1 };
        vector<P> ret;
        ret.push_back(add(a[0][0], a[1][0]));
        do {
            int d;
            if (!j[0] && !j[1]) {
                d = sgn(cross(uv(a[1][i[1]], a[1][i[1] + 1]),
                    uv(a[0][i[0]], a[0][i[0] + 1]))) >= 0;
            }
            else if (!j[0]) d = 0;
            else if (!j[1]) d = 1;
            else break;
            ret.push_back(add(uv(a[d][i[d]], a[d][i[d] + 1]), ret.back()));
            i[d] = (i[d] + 1) % len[d];
            if (i[d] == 0) j[d] = 1;
        } while (!j[0] || !j[1]);
        return ret;
        };
    for (auto& i : C) i = convex_hull(i);
    array<CI, 2> ret;ret[0] = C[0];
    for (int i = 1;i < C.size();i++) {
        ret[1] = C[i];
        ret[0] = run(ret);
    }
    return ret[0];
}

array<int, 2> cover(const vector<P>& a, const P& o) {//回转数法判断点是否在多边形内(O(n)),要求顺序是顺时针or逆时针
    int cnt = 0, n = a.size();//回转数=0表示在多边形外
    for (int i = 0; i < n; i++) {
        P u = a[i], v = a[next(i, n)];
        if (sgn(cross(uv(o, u), uv(o, v))) == 0 && sgn(dot(uv(o, u), uv(o, v))) <= 0) return { 1,1 };//在多边形上
        if (sgn(u[1] - v[1]) == 0) continue;
        if (sgn(u[1] - v[1]) < 0 && loca(u, v, o) <= 0) continue;
        if (sgn(u[1] - v[1]) > 0 && loca(u, v, o) >= 0) continue;
        if (sgn(u[1] - o[1]) < 0 && sgn(v[1] - o[1]) >= 0) cnt++;
        if (sgn(u[1] - o[1]) >= 0 && sgn(v[1] - o[1]) < 0) cnt--;
    }
    return { cnt,0 };//返回值表示回转数及是否在多边形一条边上
}
```

### 简易版计算几何（double)

``` CPP
#define debugP(x) cerr<<#x<<":("<<x[0]<<','<<x[1]<<")"<<endl
using ld = long double;
using P = array<int, 2>;
using PD = array<ld, 2>;
using LI = array<P, 2>;
using LD = array<PD, 2>;
const ld eps = 1e-9;

namespace Geometry_D {
    PD PtoPD(P u) {
        return (PD) { (ld)u[0], (ld)u[1] };
    }
    LD LItoLD(LI l) {
        LD nl;
        nl[0] = PtoPD(l[0]);
        nl[1] = PtoPD(l[1]);
        return nl;
    }
    int sgn(ld x) {
        return x > eps ? 1 : (x < -eps ? -1 : 0);
    }
    PD uv(PD u, PD v) {
        return { v[0] - u[0],v[1] - u[1] };
    }
    ld cross(PD u, PD v) {
        return u[0] * v[1] - u[1] * v[0];
    }
    ld dot(PD u, PD v) {
        return u[0] * v[0] + u[1] * v[1];
    }
    ld dis(PD u, PD v) {
        return sqrt((u[0] - v[0]) * (u[0] - v[0]) + (u[1] - v[1]) * (u[1] - v[1]));
    }

    PD mul(PD p, ld x) {
        return { p[0] * x,p[1] * x };
    }

    PD add(PD u, PD v) {
        return { u[0] + v[0],u[1] + v[1] };
    }
    PD sub(PD u, PD v) {
        return { u[0] - v[0],u[1] - v[1] };
    }
    PD div(PD p, ld x) {
        return { p[0] / x,p[1] / x };
    }
    int loca(PD u, PD v, PD w) {
        return sgn(cross(uv(u, v), uv(u, w)));
    }

    PD project(LD l, PD p) {
        PD base = uv(l[0], l[1]);//两点式描述直线
        ld r = dot(uv(l[0], p), base) / (base[0] * base[0] + base[1] * base[1]);
        return add(l[0], mul(base, r));
    }
    PD rotate(P l, double angle) {//逆时针旋转angle
        ld cosa = cos(angle), sina = sin(angle);
        return { l[0] * cosa - l[1] * sina, l[0] * sina + l[1] * cosa };
    }
    bool point_on_segment(PD a, LD line) {
        return sgn(cross(uv(a, line[0]), uv(a, line[1]))) == 0
            && sgn(dot(uv(a, line[0]), uv(a, line[1]))) <= 0;
    }

    bool inter_judge_segment(LD a, LD b) {
        //一个点的端点在另一个线段
        if (point_on_segment(b[0], a)
            || point_on_segment(b[1], a)
            || point_on_segment(a[0], b)
            || point_on_segment(a[1], b))
            return 1;
        //跨立试验
        return (loca(a[0], b[0], a[1]) * loca(a[0], b[1], a[1]) < 0
            && loca(b[0], a[0], b[1]) * loca(b[0], a[1], b[1]) < 0
            );
    }

    bool inter_judge(LD a, LD b) {
        return sgn(cross(uv(a[0], a[1]), uv(a[0], b[0]))
            - cross(uv(a[0], a[1]), uv(a[0], b[1]))) != 0;
    }
    PD line_inter(LD a, LD b) {
        ld s1 = cross(uv(a[0], a[1]), uv(a[0], b[0]));
        ld s2 = cross(uv(a[0], a[1]), uv(a[0], b[1]));
        return div(sub(mul(b[0], s2), mul(b[1], s1)), s2 - s1);
    }
    array<PD, 2> inter_circle_line(PD o, ld r, LD line) {
        PD s = line[0], t = line[1];
        PD pr = project(line, o);
        ld d = dis(o, pr);
        if (sgn(d - r) > 0) return { nan(""),nan("") };//use isnan(x) to check
        ld len = sqrt(r * r - d * d);
        PD dir = div(uv(s, t), dis(s, t));
        PD inter1 = add(pr, mul(dir, len));
        PD inter2 = add(pr, mul(dir, -len));
        return { inter1,inter2 };
    }
};
using namespace Geometry_D;
```



### Point(封装)

```c++
// *表示叉乘和数乘，&表示点乘
namespace GEO {
#define tmpl template<typename T>
    const ld eps = 1e-10;
    const ld PI = acos(-1);
    ld getld() { string x;cin >> x;return stold(x); }
    int sgn(const ld& x) { if (fabs(x) < eps) return 0;else if (x < 0)return -1;return 1; }
    int sgn(const ll& x) { if (x < 0) return -1;return x > 0; }
    ll __ceil(ld x) {
        ll k = ceil(x);
        while (k < x) k++;
        while (k > x + 1) k--;
        return k;
    }
    ll __floor(ld x) {
        ll k = floor(x);
        while (k > x) k--;
        while (k < x - 1) k++;
        return k;
    }
    namespace EXTRA {
        ld theta(ld a2, ld b2, ld c2) { return acos((a2 + b2 - c2) / (2.0 * sqrt(a2 * b2))); }//余弦定理
        ld costheta(ld a2, ld b2, ld c2) { return (a2 + b2 - c2) / (2.0 * sqrt(a2 * b2)); }//余弦定理
        ld Sabc(ld a, ld b, ld c) {//海伦公式三角形面积(三维情况下使用)
            ld p = (a + b + c) / 2;
            return sqrt(p * (p - a) * (p - b) * (p - c));
        }
        //a/sinA=2R ,正弦定理

        ld toArc(ld x) {//角度转弧度
            return PI / 180 * x;
        }
        ld toDeg(ld x) {//弧度转角度
            return x / PI * 180;
        }

    }

    tmpl struct Point {
        T x, y;
        Point() {}
        Point(T x, T y) :x(x), y(y) {}
        operator Point<ll>() const { return Point<ll>(x, y); }
        operator Point<ld>() const { return Point<ld>(x, y); }
        bool friend operator<(const Point<T>& a, const Point<T>& b) { int s = sgn(a * b);return s > 0 || s == 0 && sgn(a.len2() - b.len2()) < 0; }
        bool friend xycmp(const Point<T>& a, const Point<T>& b) { return sgn(a.x - b.x) == 0 ? sgn(a.y - b.y) < 0 : a.x < b.x; }
        bool friend operator==(const Point<T>& a, const Point<T>& b) { return sgn(a.x - b.x) == 0 && sgn(a.y - b.y) == 0; }
        bool friend operator!=(const Point<T>& a, const Point<T>& b) { return !(a == b); }
        Point<T> friend operator+(const Point<T>& a, const Point<T>& b) { return Point(a.x + b.x, a.y + b.y); }
        Point<T> friend operator-(const Point<T>& a, const Point<T>& b) { return Point(a.x - b.x, a.y - b.y); }
        Point<T> friend operator*(const T& k, const Point<T>& a) { return Point(k * a.x, k * a.y); }
        Point<T> friend operator*(const Point<T>& a, const T& k) { return Point(k * a.x, k * a.y); }
        Point<T> friend operator/(const Point<T>& a, T k) { return Point(a.x / k, a.y / k); }
        void operator+=(const Point<T>& o) { x += o.x; y += o.y; }
        void operator-=(const Point<T>& o) { x -= o.x; y -= o.y; }
        void operator*=(const T& k) { x *= k; y *= k; }
        void operator/=(const T& k) { x /= k; y /= k; }

        T friend operator*(const Point<T>& a, const Point<T>& b) { return a.x * b.y - a.y * b.x; }
        T friend operator&(const Point<T>& a, const Point<T>& b) { return a.x * b.x + a.y * b.y; }

        ld len() const { return sqrt(len2()); }//模长
        T len2() const { return x * x + y * y; }//模长的平方

        int toleft(const Point<T>& o)const { T t = (*this) * o;return (sgn(t) > 0) - (sgn(t) < 0); }
    };

    const Point<ld> no_pos = Point<ld>(INFINITY, INFINITY), all_pos = Point<ld>(-INFINITY, -INFINITY);

    istream& operator>>(istream& cin, Point<ll>& o) {
        return cin >> o.x >> o.y;
    }
    istream& operator>>(istream& cin, Point<ld>& o) {
        string s;
        cin >> s;o.x = stold(s);
        cin >> s;o.y = stold(s);
        return cin;
    }
    tmpl ostream& operator<<(ostream& cout, const Point<T>& o) {
        if ((Point<ld>)o == all_pos) return cout << "All Position";
        if ((Point<ld>)o == no_pos) return cout << "No Position";
        return cout << '(' << o.x << ',' << o.y << ')';
    }
    //坐标轴归右上象限，返回值 [1,4]
    const int DS[4] = { 1, 2, 4, 3 };
    tmpl int quad(const Point<T>& o) { return DS[(sgn(o.y) < 0) * 2 + (sgn(o.x) < 0)]; }

    tmpl bool angle_cmp(const Point<T>& a, const Point<T>& b) {
        int c = quad(a), d = quad(b);
        if (c != d) return c < d;
        return a * b > 0;
    }
    tmpl void psort(vector<Point<T>>& ps, const Point<T>& o) {
        sort(ps.begin(), ps.end(), [&](auto v1, auto v2) {return angle_cmp(v1 - o, v2 - o);});
    }

    tmpl ld distance(const Point<T>& a, const Point<T>& b) { return (a - b).len(); }
    tmpl T distance2(const Point<T>& a, const Point<T>& b) { return (a - b).len2(); }

    //向量旋转
    tmpl Point<T> rotate_right_90(const Point<T>& m) { return Point(m.y, -m.x); }//顺时针90度
    tmpl Point<T> rotate_left_90(const Point<T>& m) { return Point(-m.y, m.x); }//逆时针90度
    tmpl Point<ld> rotate(const Point<T>& l, ld angle) {//逆时针旋转angle
        ld cosa = cos(angle), sina = sin(angle);
        return { l.x * cosa - l.y * sina, l.x * sina + l.y * cosa };
    }
    tmpl Point<ld> rotate(const Point<T>& l, ld sina, ld cosa) {//逆时针旋转angle,已知sin(angel)/cos(angle)
        return { l.x * cosa - l.y * sina, l.x * sina + l.y * cosa };
    }
    Point<ld> norm(Point<ld> vec, ld r) { ld l = vec.len();if (!sgn(l)) return vec;r /= l;return { vec.x * r,vec.y * r }; }//向量改变长度

    //无向直线,切勿当向量用.向量直接用Point即可
    tmpl struct Line {
        Point<T> o, v;//点向式表示直线
        Line() {}
        Line(const Point<T>& a, const Point<T>& b, int twopoint, int norm);//两点 or 点向
        bool friend operator<(const Line<T>& a, const Line<T>& b) { int t = sgn(a.d * b.d);return t ? t > 0:a.d * a.o < b.d * b.o; }
        bool friend operator!=(const Line<T>& a, const Line<T>& b) { return !(a == b); }
        friend istream& operator>>(istream& cin, Line<T>& o) {
            return cin >> o.o >> o.d;
        }
        friend ostream& operator<<(ostream& cout, const Line<T>& o) {
            return cout << '(' << o.v.x << " k + " << o.o.x << " , " << o.v.y << " k + " << o.o.y << ")";
        }
        int toleft(const Point<T>& p) { return v.toleft(p - o); }

    };
    bool operator==(const Line<ll>& a, const Line<ll>& b) { return a.v == b.v && (b.o - a.o) * a.v == 0; }
    bool operator==(const Line<ld>& a, const Line<ld>& b) { return sgn(a.v * b.v) == 0 && sgn((b.o - a.o) * a.v) == 0; }

    template<> Line<ll>::Line(const Point<ll>& a, const Point<ll>& b, int twopoint, int norm) {
        o = a;
        v = twopoint ? b - a : b;
        ll tmp = gcd(v.x, v.y);
        assert(tmp);
        if (norm) if (v.x < 0 || v.x == 0 && v.y < 0) tmp = -tmp;
        v.x /= tmp; v.y /= tmp;
    }

    template<> Line<ld>::Line(const Point<ld>& a, const Point<ld>& b, int twopoint, int norm) {
        o = a;
        v = twopoint ? b - a : b;
        if (norm) if (sgn(v.x) < 0 || sgn(v.x) == 0 && sgn(v.y) < 0) v.x = -v.x, v.y = -v.y;
    }

    //慎用，一般用向量旋转即可
    tmpl Line<T> rotate_right_90(const Line<T>& m) { return Line(m.o, Point(m.v.y, -m.v.x), 0, 0); }//顺时针90度
    tmpl Line<T> rotate_left_90(const Line<T>& m) { return Line(m.o, Point(-m.v.y, m.v.x), 0, 0); }//逆时针90度
    tmpl Line<ld> rotate(const Line<T>& l, ld angle) {//逆时针旋转angle
        return { (Point<ld>)l.o, {l.v.x * cos(angle) - l.v.y * sin(angle), l.v.x * sin(angle) + l.v.y * cos(angle)}, 0, 0 };
    }
    tmpl Line<ld> __rotate(const Line<T>& l, ld sina, ld cosa) {//逆时针旋转angle,已知sin(angel)/cos(angle)
        return { (Point<ld>)l.o, {l.v.x * cosa - l.v.y * sina, l.v.x * sina + l.v.y * cosa}, 0, 0 };
    }

    tmpl ld get_angle(const Line<T>& a, const Line<T>& b) { return asin((a.v * b.v) / (a.v.len() * b.v.len())); }

    tmpl Point<ld> intersect(const Line<T>& a, const Line<T>& b) {
        if (sgn(a.v * b.v) == 0) {
            if (sgn(a.v * (b.o - a.o)) == 0) return all_pos;//重合
            return no_pos;//平行
        }
        return (Point<ld>)a.o + (b.o - a.o) * b.v / (ld)(a.v * b.v) * (Point<ld>)a.v;
    }
    tmpl ld distance(const Line<T>& l, const Point<T>& o) { return abs(l.v * (o - l.o) / l.v.len()); }
    tmpl ld distance(const Point<T>& o, const Line<T>& l) { return abs(l.v * (o - l.o) / l.v.len()); }

    tmpl Point<ld> proj(const Line<T>& l, const Point<T>& o) {
        auto newo = (Point<ld>)o, newlo = (Point<ld>)l.o, newlv = (Point<ld>)l.v;
        ld k = (newo - newlo) & newlv;k /= newlv & newlv;return newlo + k * newlv;
    }//投影

    tmpl Point<ld> proj(const Point<T>& o, const Line<T>& l) {
        auto newo = (Point<ld>)o, newlo = (Point<ld>)l.o, newlv = (Point<ld>)l.v;
        ld k = (newo - newlo) & newlv;k /= newlv & newlv;return newlo + k * newlv;
    }//投影

    struct Circle {
        Point<ld> o;//圆心
        ld r;//半径
        ld area() { return PI * r * r; }
        ld circum() { return 2 * PI * r; }
        Circle() {}
        Circle(const Point<ld>& o, ld r = 0) :o(o), r(r) {}
        Circle(const Point<ld>& a, const Point<ld>& b) {
            o = (a + b) * 0.5;
            r = distance(b, o);
        }

        Circle(const Point<ld>& a, const Point<ld>& b, const Point<ld>& c) {//三点构造外接圆（非最小圆）
            auto A = (b + c) * 0.5, B = (a + c) * 0.5;
            o = intersect(rotate_right_90(Line(A, c, 1, 1)), rotate_right_90(Line(B, c, 1, 1)));
            r = distance(o, c);
        }

        Circle(const vector<Point<ll>>& b) {
            vector<Point<ld>> a(b.size());
            int n = a.size();
            for (int i = 0; i < a.size(); i++) a[i] = (Point<ld>)b[i];
            *this = Circle(a);
        }

        Circle(vector<Point<ld>> a) {//最小圆覆盖（随机增量法）
            int n = a.size();
            mt19937 rnd(75643);
            shuffle(a.begin(), a.end(), rnd);
            *this = Circle(a[0]);
            for (int i = 1; i < n; i++) {
                if (!cover(a[i])) {
                    *this = Circle(a[i]);
                    for (int j = 0; j < i; j++) {
                        if (!cover(a[j])) {
                            *this = Circle(a[i], a[j]);
                            for (int k = 0; k < j; k++) {
                                if (!cover(a[k])) *this = Circle(a[i], a[j], a[k]);
                            }
                        }
                    }
                }
            }
        }

        tmpl bool cover(const Point<T>& a) { return sgn(distance((Point<ld>)a, o) - r) <= 0; }

        bool friend operator==(Circle a, Circle b) { return (a.o == b.o) && sgn(a.r - b.r) == 0; }
        bool friend operator < (Circle a, Circle b) { return ((a.o < b.o) || ((a.o == b.o) && sgn(a.r - b.r) < 0)); }

    };

    tmpl struct Segment {
        Point<T> s, e;
        Segment() { }
        Segment(Point<T> a, Point<T> b) {
            int t = sgn(a.x - b.x);
            if (t > 0 || t == 0 && a.y > b.y) swap(a, b);
            s = a, e = b;
        }
    };

    tmpl bool intersect(const Segment<T>& m, const Segment<T>& n) {
        auto a = n.e - n.s, b = m.e - m.s;
        auto d = n.s - m.s;
        if (sgn(n.e.x - m.s.x) < 0 || sgn(m.e.x - n.s.x) < 0) return 0;
        if (sgn(max(n.s.y, n.e.y) - min(m.s.y, m.e.y)) < 0 || sgn(max(m.s.y, m.e.y) - min(n.s.y, n.e.y)) < 0) return 0;
        return sgn(d * b) * sgn((n.e - m.s) * b) <= 0 && sgn(d * a) * sgn((n.s - m.e) * a) <= 0;
    }

    tmpl bool on_seg(const Point<T>& p, const Segment<T> seg) { return sgn((p - seg.s) * (seg.e - seg.s)) == 0 && sgn((seg.s - p) & (seg.e - p)) <= 0; }//点在线段上
    tmpl bool on_seg(const Segment<T> seg, const Point<T>& p) { return sgn((p - seg.s) * (seg.e - seg.s)) == 0 && sgn((seg.s - p) & (seg.e - p)) <= 0; }//点在线段上

    tmpl struct Polygon {

        vector<Point<T>> p;
        struct Convex;
        Polygon() { }
        Polygon(const vector<Point<T>>& a) :p(a) {}
        ld peri() {//周长
            int n = p.size();
            ld C = (p[n - 1] - p[0]).len();
            for (int i = 1; i < n; i++) C += (p[i - 1] - p[i]).len();
            return C;
        }
        ld area() { return area2() * 0.5; }//面积
        T area2() {//两倍面积
            int n = p.size();
            T S = p[n - 1] * p[0];
            for (int i = 1; i < n; i++) S += p[i - 1] * p[i];
            return abs(S);
        }
        ld diam() { return sqrt(diam2()); }//直径
        T diam2() {//直径平方
            T r = 0;
            int n = p.size();
            if (n <= 2) {
                for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) r = max(r, dis2(p[i], p[j]));
                return r;
            }
            p.push_back(p[0]);
            for (int i = 0, j = 1; i < n; i++) {
                while ((p[i + 1] - p[i]) * (p[j] - p[i]) <= (p[i + 1] - p[i]) * (p[j + 1] - p[i])) if (++j == n) j = 0;
                r = max({ r, dis2(p[i], p[j]), dis2(p[i + 1], p[j]) });
            }
            p.pop_back();
            return r;
        }
        array<int, 2> cover(const Point<T>& o) const {//回转数法判断点是否在多边形内(O(n)),要求顺序是顺时针or逆时针
            int cnt = 0, n = p.size();//回转数=0表示在多边形外
            for (int i = 0; i < n; i++) {
                Point u = p[i], v = ((i == n - 1) ? p[0] : p[i + 1]);
                if (sgn((u - o) * (v - o)) == 0 && sgn((u - o) & (v - o)) <= 0) return { 1,1 };//在多边形上
                if (sgn(u.y - v.y) == 0) continue;
                Point<T> uv = v - u;
                if (sgn(u.y - v.y) < 0 && uv.toleft(o - u) <= 0) continue;
                if (sgn(u.y - v.y) > 0 && uv.toleft(o - u) >= 0) continue;
                if (sgn(u.y - o.y) < 0 && sgn(v.y - o.y) >= 0) cnt++;
                if (sgn(u.y - o.y) >= 0 && sgn(v.y - o.y) < 0) cnt--;
            }
            return { cnt,0 };//返回值表示回转数及是否在多边形一条边上
        }

        bool __cover(const Point<T>& o) {//判断点是否在多边形内(O(n))，不需要管顺序但需要把每条边的两个点存下，精度略低因为用了除法.
            int n = p.size();
            bool flag = false;
            Point<T> u, v; //多边形一条边的两个顶点
            for (int i = 0, j = n - 1;i < n;j = i++) {//枚举两条边的顶点，这里默认有序
                u = p[i], v = p[j];
                Segment<T> uv(u, v);
                if (on_seg(uv, o)) return true; //点在多边形上
                if ((sgn(u.y - o.y) > 0 != sgn(v.y - o.y) > 0) && sgn(o.x - (ld)(o.y - u.y) * (u.x - v.x) / (u.y - v.y) - u.x) < 0) flag = !flag;
            }
            return flag;
        }

        Polygon<T> operator+(const Polygon<T>& A) const {
            int n = p.size(), m = A.p.size();
            vector<Point<T>> c;
            if (min(n, m) <= 2) {
                c.reserve(n * m);
                for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) c.push_back(p[i] + A.p[j]);
                return Polygon<T>(c);
            }
            Point<T> a[n], b[m];
            for (int i = 0; i + 1 < n; i++) a[i] = p[i + 1] - p[i];
            a[n - 1] = p[0] - p[n - 1];
            for (int i = 0; i + 1 < m; i++) b[i] = A.p[i + 1] - A.p[i];
            b[m - 1] = A.p[0] - A.p[m - 1];
            c.reserve(n + m);
            c.push_back(p[0] + A.p[0]);
            int i = 0, j = 0;
            while (i < n && j < m) c.push_back(c.back() + (a[i] * b[j] > 0 ? a[i++] : b[j++]));
            while (i < n - 1) c.push_back(c.back() + a[i++]);
            while (j < m - 1) c.push_back(c.back() + b[j++]);
            return Polygon<T>(c);
        }
        void operator+=(const Polygon& a) { *this = *this + a; }
    };

    tmpl struct Polygon<T>::Convex : public Polygon<T> {
        Convex() {}
        Convex(vector<Point<T>> a) {//Andrew算法求凸包
            vector<Point<T>> stk;
            sort(a.begin(), a.end(), [&](auto& u, auto& v) {return xycmp(u, v);});
            const auto check = [&](const vector<Point<T>>& stk, const Point<T>& u) {
                const auto back1 = stk.back(), back2 = *(prev(stk.end(), 2));
                return (back1 - back2).toleft(u - back2) <= 0;
                };
            for (const auto& u : a) {
                while (stk.size() > 1 && check(stk, u)) stk.pop_back();
                stk.push_back(u);
            }
            int sz = stk.size();
            a.pop_back();reverse(a.begin(), a.end());
            for (const auto& u : a) {
                while (stk.size() > sz && check(stk, u)) stk.pop_back();
                stk.push_back(u);
            }
            if (stk.size() > 1) stk.pop_back();
            p = stk;
        }

        Convex(vector<Point<T>> a, int op) {//极角排序求凸包
            int n = a.size(), i;
            if (!n) return;
            p = a;
            for (i = 1; i < n; i++) if (p[i].x < p[0].x || p[i].x == p[0].x && p[i].y < p[0].y) swap(p[0], p[i]);
            a.resize(0); a.reserve(n);
            for (i = 1; i < n; i++) if (p[i] != p[0]) a.push_back(p[i] - p[0]);
            sort(a.begin(), a.end());
            for (i = 0; i < a.size(); i++) a[i] += p[0];
            Point<T>* st = p.data() - 1;
            int tp = 1;
            for (auto& v : a) {
                while (tp > 1 && sgn((st[tp] - st[tp - 1]) * (v - st[tp - 1])) <= 0) --tp;
                st[++tp] = v;
            }
            p.resize(tp);
        }

        bool cover(const Point<T>& o) const {//点是否在凸包内(O(logn))
            if (o.x < p[0].x || o.x == p[0].x && o.y < p[0].y) return 0;
            if (o == p[0]) return 1;
            if (p.size() == 1) return 0;
            ll tmp = (o - p[0]) * (p.back() - p[0]);
            if (tmp == 0) return distance2(o, p[0]) <= distance2(p.back(), p[0]);
            if (tmp < 0 || p.size() == 2) return 0;
            int x = upper_bound(p.begin() + 1, p.end(), o, [&](const Point<T>& a, const Point<T>& b) { return (a - p[0]) * (b - p[0]) > 0; }) - p.begin() - 1;
            return (o - p[x]) * (p[x + 1] - p[x]) <= 0;
        }
    };



#undef tmpl
}

using namespace GEO;using namespace EXTRA;
template<typename T> using Convex = typename Polygon<T>::Convex;
template<typename T> using Vector = Point<T>;
```





### 自适应辛普森

```c++
const ld eps = 1e-10;
//积分上下限
ld a, b;
cin >> a >> b;
//函数f(x)的表达式
auto f = [&](ld x)->ld {
    return sin(x) / x;
};

auto simpson = [&](ld l, ld r)->ld {
    return (r - l) * (f(l) + f(r) + 4 * f((l + r) / 2)) / 6;
};

auto calc = [&](auto&& calc, ld l, ld r, ld res)->ld {
    ld m = (l + r) / 2, a = simpson(l, m), b = simpson(m, r);
    if (abs(a + b - res) < eps) return res;
    return calc(calc, l, m, a) + calc(calc, m, r, b);
};
//如果积分下限为0,则把a替换成eps. 
//如果积分上限是无穷,则替换成一个精度足够的数字使得答案正确
cout << calc(calc, a, b, simpson(a, b)) << endl;
```



## 杂项/DP

### 求超集中数组f的和

``` cpp
for(int j = 0; j < n; j++) 
    for(int i = 0; i < 1 << n; i++)
        if(!(i >> j & 1)) f[i] += f[i ^ (1 << j)];
```

### 求某个字符串s[i-n]以及s[j-n]的lcp（最长公共前缀）

``` cpp
for(int i=s1;i>=1;--i) {
        for(int j=s1;j>=i+1;--j) {
            if (s[i]==s[j]) lcp[i][j]=lcp[i+1][j+1]+1;
            else lcp[i][j]=0;
            lcp[i][j]=min(lcp[i][j],j-i);
        }
    }
```

### 快速离散化

``` cpp
#include <bits/stdc++.h>
using namespace std;

//基数排序离散化
namespace Discretization_Int {
    const int base = (1 << 17) - 1;
    vector<int> c(base + 10);
    vector<pair<int, int>> data, tmp;
    void discretization(vector<int>& input) {
        int n = input.size();
        data.resize(n);
        tmp.resize(n);
        for (int i = 0; i < n; i++) data[i] = { input[i], i };
        for (int i = 0; i < 32; i += 16) {
            fill(c.begin(), c.end(), 0);
            for (int j = 0; j < n; j++) c[(data[j].first >> i) & base]++;
            for (int j = 1; j <= base; ++j) c[j] += c[j - 1];
            for (int j = n - 1; j >= 0; --j) tmp[--c[(data[j].first >> i) & base]] = data[j];
            data.swap(tmp);
        }

        for (int i = 0, j = -1; i < n; i++) {
            if (i == 0 || data[i].first != data[i - 1].first) ++j;
            input[data[i].second] = j;
        }
    }
}

namespace Discretization_LL {
    const int base = (1 << 17) - 1;
    vector<int> c(base + 10);
    vector<pair<long long, int>> data, tmp;

    void discretization(vector<long long>& input) {
        int n = input.size();
        data.resize(n);
        tmp.resize(n);

        for (int i = 0; i < n; i++)
            data[i] = { input[i], i };

        for (int i = 0; i < 64; i += 16) {
            fill(c.begin(), c.end(), 0);
            for (int j = 0; j < n; ++j) c[(data[j].first >> i) & base]++;
            for (int j = 1; j <= base; j++) c[j] += c[j - 1];
            for (int j = n - 1; j >= 0; j--) tmp[--c[(data[j].first >> i) & base]] = data[j];
            data.swap(tmp);
        }

        for (int i = 0, j = -1; i < n; i++) {
            if (i == 0 || data[i].first != data[i - 1].first) ++j;
            input[data[i].second] = j;
        }
    }
}


int main() {
    int n = 1000000;
    default_random_engine e;
    uniform_int_distribution<long long> d(0, LLONG_MAX);
    e.seed(time(0));
    vector<long long> A(n), B(n), tmp(n);
    for (int i = 0; i < n; ++i)
        A[i] = B[i] = tmp[i] = d(e);
    printf("start....\n");
    auto start = clock();
    sort(tmp.begin(), tmp.end());
    int sz = unique(tmp.begin(), tmp.end()) - tmp.begin();
    for (int i = 0; i < n; ++i)
        A[i] = lower_bound(tmp.begin(), tmp.begin() + sz, A[i]) - tmp.begin();
    printf("std::sort: %f\n", static_cast<double>(clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    Discretization_LLong::discretization(B);
    printf("std::sort: %f\n", static_cast<double>(clock() - start) / CLOCKS_PER_SEC);
    for (int i = 0; i < n; ++i)
        if (A[i] != B[i])
            abort();
    return 0;
}
```



### 质数模数自动取模

```cpp
template <typename T>
concept Can_bit = requires(T x) { x >>= 1; };

template <int MOD>
struct modint {
    int val;
    static int norm(const int& x) { return x < 0 ? x + MOD : x; }
    static constexpr int get_mod() { return MOD; }
    modint inv() const {
        assert(val);  // 确保val不为0
        int a = val, b = MOD, u = 1, v = 0, t;
        while (b > 0) t = a / b, swap(a -= t * b, b), swap(u -= t * v, v);
        assert(a == 1);  // 确保a和MOD互质
        return modint(u);
    }
    modint() : val(0) {}
    modint(const int& m) : val(norm(m)) {}
    modint(const long long& m) : val(norm(m % MOD)) {}
    modint operator-() const { return modint(norm(-val)); }
    bool operator==(const modint& o) const { return val == o.val; }
    bool operator<(const modint& o) const { return val < o.val; }
    modint& operator+=(const modint& o) { return val = (1ll * val + o.val) % MOD, *this; }
    modint& operator-=(const modint& o) { return val = norm(1ll * val - o.val), *this; }
    modint& operator*=(const modint& o) { return val = static_cast<int>(1ll * val * o.val % MOD), *this; }
    modint& operator/=(const modint& o) { return *this *= o.inv(); }
    modint& operator^=(const modint& o) { return val ^= o.val, *this; }
    modint& operator>>=(const modint& o) { return val >>= o.val, *this; }
    modint& operator<<=(const modint& o) { return val <<= o.val, *this; }
    modint operator-(const modint& o) const { return modint(*this) -= o; }
    modint operator+(const modint& o) const { return modint(*this) += o; }
    modint operator*(const modint& o) const { return modint(*this) *= o; }
    modint operator/(const modint& o) const { return modint(*this) /= o; }
    modint operator^(const modint& o) const { return modint(*this) ^= o; }
    modint operator>>(const modint& o) const { return modint(*this) >>= o; }
    modint operator<<(const modint& o) const { return modint(*this) <<= o; }
    friend std::istream& operator>>(std::istream& is, modint& a) {
        long long v;
        return is >> v, a.val = norm(v % MOD), is;
    }
    friend std::ostream& operator<<(std::ostream& os, const modint& a) { return os << a.val; }
    friend std::string tostring(const modint& a) { return std::to_string(a.val); }
    template <Can_bit T>
    friend modint qpow(const modint& a, const T& b) {
        assert(b >= 0);
        modint x = a, res = 1;
        for (T p = b; p; x *= x, p >>= 1)
            if (p & 1) res *= x;
        return res;
    }
};//modint<1000000007>md(100)
```



### 变模数取模

```cpp
struct barrett {
    unsigned int _m;
    unsigned long long im;

    // @param m `1 <= m`
    explicit barrett(unsigned int m) : _m(m), im((unsigned long long)(-1) / m + 1) {}

    // @return m
    unsigned int umod() const { return _m; }

    // @param a `0 <= a < m`
    // @param b `0 <= b < m`
    // @return `a * b % m`
    unsigned int mul(unsigned int a, unsigned int b) const {
        // [1] m = 1
        // a = b = im = 0, so okay

        // [2] m >= 2
        // im = ceil(2^64 / m)
        // -> im * m = 2^64 + r (0 <= r < m)
        // let z = a*b = c*m + d (0 <= c, d < m)
        // a*b * im = (c*m + d) * im = c*(im*m) + d*im = c*2^64 + c*r + d*im
        // c*r + d*im < m * m + m * im < m * m + 2^64 + m <= 2^64 + m * (m + 1) < 2^64 * 2
        // ((ab * im) >> 64) == c or c + 1
        unsigned long long z = a;
        z *= b;
#ifdef _MSC_VER
        unsigned long long x;
        _umul128(z, im, &x);
#else
        unsigned long long x =
            (unsigned long long)(((unsigned __int128)(z)*im) >> 64);
#endif
        unsigned long long y = x * _m;
        return (unsigned int)(z - y + (z < y ? _m : 0));
    }
};

```

### 本质不同子序列个数

```cpp
vector<int> fore(26);
        int cur = 1;
        for (int i = 1; i < =n; i++)
        {
            int nowcount = cur;
            cur = ((cur * 2 % mod) - fore[s[i] - 'a'] + mod) % mod;
            fore[s[i] - 'a'] = nowcount;
        }
```

### 找递推式

``` cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
#define int long long 
#define endl '\n'

const int MOD = 1e9 + 7;
// k 为 m 最高次数 且　a[m] == 1
namespace BerlekampMassey {
    inline void up(ll& a, ll b) { (a += b) %= MOD; }

    vector<ll> mul(const vector<ll>& a, const vector<ll>& b, const vector<ll>& m, int k) {
        vector<ll> r; r.resize(2 * k - 1);
        for (int i = 0;i < k;i++)
            for (int j = 0;j < k;j++)
                up(r[i + j], a[i] * b[j]);
        for (int i = k - 2;i >= 0;i--) {
            for (int j = 0;j < k;j++)
                up(r[i + j], r[i + k] * m[j]);
            r.pop_back();
        }
        return r;
    }
    ll pow_mod(ll x, ll y) {
        ll ret = 1;
        for (;y;y >>= 1) { if (y & 1) ret = ret * x % MOD;x = x * x % MOD; }
        return ret;
    }
    ll get_inv(ll x, ll MOD) {
        return pow_mod(x, MOD - 2);
    }
    vector<ll> pow(ll n, const vector<ll>& m) {
        int k = (int)m.size() - 1; assert(m[k] == -1 || m[k] == MOD - 1);
        vector<ll> r(k), x(k); r[0] = x[1] = 1;
        for (; n; n >>= 1, x = mul(x, x, m, k))
            if (n & 1) r = mul(x, r, m, k);
        return r;
    }
    ll go(const vector<ll>& a, const vector<ll>& x, ll n) {
        // a: (-1, a1, a2, ..., ak).reverse
        // x: x1, x2, ..., xk
        // x[n] = sum[a[i]*x[n-i],{i,1,k}]
        int k = (int)a.size() - 1;
        if (n <= k) return x[n - 1];
        vector<ll> r = pow(n - 1, a);
        ll ans = 0;
        for (int i = 0;i < k;i++)
            up(ans, r[i] * x[i]);
        return ans;
    }

    vector<ll> BM(const vector<ll>& x) {
        vector<ll> a = { -1 }, b = { 233 };
        for (int i = 1;i < x.size();i++) {
            b.push_back(0);
            ll d = 0, la = a.size(), lb = b.size();
            for (int j = 0;j < la;j++) up(d, a[j] * x[i - la + 1 + j]);
            if (d == 0) continue;
            vector<ll> t; for (auto& v : b) t.push_back(d * v % MOD);
            for (int j = 0;j < a.size();j++) up(t[lb - 1 - j], a[la - 1 - j]);
            if (lb > la) {
                b = a;
                ll inv = -get_inv(d, MOD);
                for (auto& v : b) v = v * inv % MOD;
            }
            a.swap(t);
        }
        for (auto& v : a) up(v, MOD);
        return a;
    }
}

void GET(const vector<ll>& x) {
    vector<ll> a = BerlekampMassey::BM(x);
    cout << "a[n] = ";
    for (int i = 0;i < a.size() - 2;i++) {
        cout << a[i] << "*a[n-" << a.size() - 1 - i << "] + ";
    }
    cout << a[a.size() - 2] << "*a[n-1]" << endl;
}

void Prework() {

}
void Solve() {
    vector<int> a = { 1,5,14,30,55,91,140,204,285,385 };
    GET(a);
}
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);
    int T = 1;
    //cin >> T;
    Prework();
    while (T--) Solve();
}
```

### 取模还原分数

``` CPP
array<int, 2> approx(int p, int q, int A) {//模数p,还原q,分子最大值A
    int x = q, y = p, a = 1, b = 0;
    while (x > A) {
        swap(x, y);swap(a, b);
        a -= x / y * b;
        x %= y;
    }
    return { x,a };//q = x/a(mod p)
}
```



### 三维偏序

```cpp
#include <algorithm>
#include <cstdio>

const int maxN = 1e5 + 10;
const int maxK = 2e5 + 10;

int n, k;

struct Element {
  int a, b, c;
  int cnt;
  int res;

  bool operator!=(Element other) {
    if (a != other.a) return true;
    if (b != other.b) return true;
    if (c != other.c) return true;
    return false;
  }
};

Element e[maxN];
Element ue[maxN];
int m, t;
int res[maxN];

struct BinaryIndexedTree {
  int node[maxK];

  int lowbit(int x) { return x & -x; }

  void Add(int pos, int val) {
    while (pos <= k) {
      node[pos] += val;
      pos += lowbit(pos);
    }
    return;
  }

  int Ask(int pos) {
    int res = 0;
    while (pos) {
      res += node[pos];
      pos -= lowbit(pos);
    }
    return res;
  }
} BIT;

bool cmpA(Element x, Element y) {
  if (x.a != y.a) return x.a < y.a;
  if (x.b != y.b) return x.b < y.b;
  return x.c < y.c;
}

bool cmpB(Element x, Element y) {
  if (x.b != y.b) return x.b < y.b;
  return x.c < y.c;
}

void CDQ(int l, int r) {
  if (l == r) return;
  int mid = (l + r) / 2;
  CDQ(l, mid);
  CDQ(mid + 1, r);
  std::sort(ue + l, ue + mid + 1, cmpB);
  std::sort(ue + mid + 1, ue + r + 1, cmpB);
  int i = l;
  int j = mid + 1;
  while (j <= r) {
    while (i <= mid && ue[i].b <= ue[j].b) {
      BIT.Add(ue[i].c, ue[i].cnt);
      i++;
    }
    ue[j].res += BIT.Ask(ue[j].c);
    j++;
  }
  for (int k = l; k < i; k++) BIT.Add(ue[k].c, -ue[k].cnt);
  return;
}

int main() {
  scanf("%d%d", &n, &k);
  for (int i = 1; i <= n; i++) scanf("%d%d%d", &e[i].a, &e[i].b, &e[i].c);
  std::sort(e + 1, e + n + 1, cmpA);
  for (int i = 1; i <= n; i++) {
    t++;
    if (e[i] != e[i + 1]) {
      m++;
      ue[m].a = e[i].a;
      ue[m].b = e[i].b;
      ue[m].c = e[i].c;
      ue[m].cnt = t;
      t = 0;
    }
  }
  CDQ(1, m);
  for (int i = 1; i <= m; i++) res[ue[i].res + ue[i].cnt - 1] += ue[i].cnt;
  for (int i = 0; i < n; i++) printf("%d\n", res[i]);
  return 0;
}
```



### 数位dp模板

```c++
memset(f, -1, sizeof f);
auto dp = [&](int x)->ll {
    auto a = to_string(x);
    reverse(a.begin(), a.end());
    int n = a.size();
    auto dfs = [&](auto&& dfs, int pos, int lead, int lim)->ll {
        if (pos == -1) return;
        int t = f[pos];
        if (!lim && !lead && t != -1) return t;
        t = 0;
        int mx = lim ? a[pos] - '0' : 9;
        if (lead) t += dfs(dfs, pos - 1, 1, 0);
        for (int i = lead;i <= mx;i++) {
            t += dfs(dfs, pos - 1, 0, lim && (i == mx));
        }
        if (!lim && !lead) f[pos] = t;
        return t;
        };
    int res = 0;
    res += dfs(dfs, n - 1, 1, 1);
    return res;
    };
```



### 状压DP子集枚举技巧

```
for (int S=1; S<(1<<n); ++S){
    for (int S0=S; S0; S0=(S0-1)&S)
        //do something.
}
```

### 树上背包模板

```cpp
int dfs(int u) {
  int p = 1;
  f[u][1] = s[u];
  for (auto v : e[u]) {
    int siz = dfs(v);
    // 注意下面两重循环的上界和下界
    // 只考虑已经合并过的子树，以及选的课程数超过 m+1 的状态没有意义
    for (int i = min(p, m + 1); i; i--)
      for (int j = 1; j <= siz && i + j <= m + 1; j++)
        f[u][i + j] = max(f[u][i + j], f[u][i] + f[v][j]);  // 转移方程
    p += siz;
  }
  return p;
}
```



### fread快读

```c++
char In[1 << 20], * ss = In, * tt = In;
#define getchar() (ss == tt && (tt = (ss = In) + fread(In, 1, 1 << 20, stdin), tt == ss) ? EOF : *ss++)

int read(char ch = 0) {
    int x = 0, f = 1;
    while (ch < '0' || ch > '9') f = ch == '-' ? -1 : 1, ch = getchar();
    while (ch >= '0' && ch <= '9') x = x * 10 + ch - 48, ch = getchar();
    return x * f;
}
char getch(char ch = 0) {
    while (ch < 'A' || (ch > 'Z' && ch < 'a') || ch > 'z') ch = getchar();
    return ch;
}
inline void write(ll x) {
    if (x < 0)putchar('-'), x = -x;
    if (x > 9)write(x / 10ll);
    putchar(x % 10 + 48);
}
```

### 约瑟夫环

``` cpp
//F(n,k) = (F(n - 1, k) + k) % n
auto josephu = [&](int n, int k) {
    int f = 0;
    for (int i = 2;i <= n;i++) {
        f = (f + k) % i;
    }
    return f;
    };//k>=n时使用

auto josephu2 = [&](auto&& josephu2, int n, int k) {
    if (n == 1) return 0;
    if (k == 1) return n - 1;
    if (k > n) return (josephus2(josephu2, n - 1, k) + k) % n;
    int f = josephus2(josephu2, n - n / k, k) - n % k;
    return f + (f < 0 ? n : (f / (k - 1)));
    };//k<n时使用，会比较快
```



### 高精度

```c++
const int MOD = 998244353;//NTT模数

int Add(int x, int y) { return (x + y >= MOD) ? x + y - MOD : x + y; }
int Dec(int x, int y) { return (x - y < 0) ? x - y + MOD : x - y; }
int mul(int x, int y) { return 1ll * x * y % MOD; }
uint qp(uint a, int b) { uint res = 1; for (; b; b >>= 1, a = mul(a, a))  if (b & 1)  res = mul(res, a); return res; }

namespace NTT {

    int sz;
    uint w[2500005], w_mf[2500005];
    int mf(int x) { return (1ll * x << 32) / MOD; }
    void init(int n) {
        for (sz = 2; sz < n; sz <<= 1);
        uint pr = qp(3, (MOD - 1) / sz);
        w[sz / 2] = 1; w_mf[sz / 2] = mf(1);
        for (int i = 1; i < sz / 2; i++)  w[sz / 2 + i] = mul(w[sz / 2 + i - 1], pr), w_mf[sz / 2 + i] = mf(w[sz / 2 + i]);
        for (int i = sz / 2 - 1; i; i--)  w[i] = w[i << 1], w_mf[i] = w_mf[i << 1];
    }
    void ntt(vector<uint>& A, int L) {
        for (int d = L >> 1; d; d >>= 1)
            for (int i = 0; i < L; i += (d << 1))
                for (int j = 0; j < d; j++) {
                    uint x = A[i + j] + A[i + d + j];
                    if (x >= 2 * MOD)  x -= 2 * MOD;
                    ll t = A[i + j] + 2 * MOD - A[i + d + j], q = t * w_mf[d + j] >> 32; int y = t * w[d + j] - q * MOD;
                    A[i + j] = x; A[i + d + j] = y;
                }
        for (int i = 0; i < L; i++)  if (A[i] >= MOD)  A[i] -= MOD;
    }
    void intt(vector<uint>& A, int L) {
        for (int d = 1; d < L; d <<= 1)
            for (int i = 0; i < L; i += (d << 1))
                for (int j = 0; j < d; j++) {
                    uint x = A[i + j]; if (x >= 2 * MOD)  x -= 2 * MOD;
                    ll t = A[i + d + j], q = t * w_mf[d + j] >> 32, y = t * w[d + j] - q * MOD;
                    A[i + j] = x + y; A[i + d + j] = x + 2 * MOD - y;
                }
        int k = (L & (-L));
        reverse(A.begin() + 1, A.end());
        for (int i = 0; i < L; i++) {
            ll m = -A[i] & (L - 1);
            A[i] = (A[i] + m * MOD) / k;
            if (A[i] >= MOD)  A[i] -= MOD;
        }
    }
}

struct bigint {
    vector<int> nums;
    int operator[](const int& k)const { return nums[k]; }
    int& operator[](const int& k) { return nums[k]; }
    int size() { return nums.size(); }
    void push_back(int x) { nums.push_back(x); }
    bigint(int x = 0) {
        do {
            nums.push_back(x % 10);
            x /= 10;
        } while (x);
    }

    bigint(string s) {
        for (int i = s.size() - 1; i >= 0; i--)
            nums.push_back(s[i] - '0');
        trim();
    }

    void trim() {
        while (nums.size() > 1 && nums.back() == 0) {
            nums.pop_back();
        }
    }

    void clear() {
        nums.clear();
    }

    friend istream& operator>>(istream& cin, bigint& num) {
        string tnum;
        cin >> tnum;
        num = tnum;
        return cin;
    }
    friend ostream& operator<<(ostream& cout, bigint num) {
        bool start = false;
        for (int i = num.size() - 1; i >= 0; i--) {
            if (!start && num[i] == 0)
                continue;
            start = true;
            cout << num[i];
        }
        if (!start)
            cout << 0;
        return cout;
    }
};

bool operator<(bigint a, bigint b) {
    if (a.size() != b.size())
        return a.size() < b.size();
    for (int i = a.size() - 1; i >= 0; i--)
        if (a[i] != b[i])
            return a[i] < b[i];
    return false;
}

bool operator>(bigint a, bigint b) {
    return b < a;
}

bool operator<=(bigint a, bigint b) {
    return !(a > b);
}

bool operator>=(bigint a, bigint b) {
    return !(a < b);
}

bool operator==(bigint a, bigint b) {
    return !(a < b) && !(a > b);
}

bool operator!=(bigint a, bigint b) {
    return a < b || a > b;
}

bigint operator+(bigint a, bigint b) {
    bigint res;
    res.clear();
    int t = 0;
    int mx = max(a.size(), b.size());
    for (int i = 0; i < mx || t; i++) {
        if (i < a.size()) {
            t += a[i];
        }
        if (i < b.size()) {
            t += b[i];
        }
        res.push_back(t % 10);
        t /= 10;
    }
    res.trim();
    return res;
}

bigint operator-(bigint a, bigint b) {
    bigint res(a);
    bigint sub(b);
    int flag = 0;
    int len = res.size();
    while (sub.size() < res.size())
        sub.push_back(0);
    for (int i = 0; i < len; i++) {
        if (res[i] + flag >= sub[i]) {
            res[i] = res[i] + flag - sub[i];
            flag = 0;
        }
        else {
            res[i] = res[i] + 10 + flag - sub[i];
            flag = -1;
        }
    }
    res.trim();
    return res;
}

// bigint operator*(bigint a, bigint b) {//n^2
//     bigint res;
//     res.resize(a.size() + b.size(), 0);
//     for (int i = 0; i < a.size(); i++) {
//         for (int j = 0; j < b.size(); j++) {
//             res[i + j] += a[i] * b[j];
//             res[i + j + 1] += res[i + j] / 10;
//             res[i + j] %= 10;
//         }
//     }
//     res.trim();
//     return res;
// }

bigint operator*(bigint a, bigint b) {//nlogn
    bigint res;res.nums.pop_back();
    int dega = a.size() - 1, degb = b.size() - 1;
    int n = dega + degb + 1;
    int lim;for (lim = 1; lim < n; lim <<= 1); NTT::init(lim);
    vector<uint> A(lim); for (int i = 0;i <= dega;i++) A[i] = a[i];
    vector<uint> B(lim);for (int i = 0;i <= degb;i++) B[i] = b[i];
    NTT::ntt(A, lim);NTT::ntt(B, lim);
    for (int i = 0;i < lim;i++) A[i] = mul(A[i], B[i]);
    NTT::intt(A, lim);
    for (int i = 0, t = 0;i < lim || t;i++) {
        if (i < lim) t += A[i];
        res.push_back(t % 10);t /= 10;
    }
    res.trim();
    return res;
}

bigint operator*(bigint a, ll b) {
    bigint res(a);
    int carry = 0;
    for (int i = 0; i < a.size(); i++) {
        carry += a[i] * b;
        res[i] = carry % 10;
        carry /= 10;
    }
    while (carry > 0) {
        res.push_back(carry % 10);
        carry /= 10;
    }
    //res.trim();
    return res;
}

bigint operator/(bigint a, bigint b) {
    bigint tnum(a);
    if (a < b)
        return 0;
    int n = a.size() - b.size();
    b.nums.insert(b.nums.begin(), n, 0);
    if (tnum >= b) {
        n++;
        b.nums.insert(b.nums.begin(), 0);
    }
    bigint ans;
    ans.nums.assign(n, 0);
    int n2 = b.size();
    while (n--) {
        n2--;
        b.nums.erase(b.nums.begin());
        while (!(tnum < b)) {
            int n1 = tnum.size();
            for (int j = 0; j < n2; j++) {
                tnum[j] -= b[j];
                if (tnum[j] < 0) {
                    tnum[j + 1]--;
                    tnum[j] += 10;
                }
            }
            tnum.trim();
            ans[n]++;
        }
    }
    ans.trim();
    return ans;
}

bigint operator/(bigint a, ll b) {
    bigint ans;
    ans.clear();
    int r = 0;
    for (int i = a.size() - 1; i >= 0; i--) {
        r = r % b * 10 + a[i];
        ans.push_back(r / b);
    }
    reverse(ans.nums.begin(), ans.nums.end());
    ans.trim();
    return ans;
}

bigint operator%(bigint a, bigint b) {
    bigint div_res = a / b;
    return a - div_res * b;
}

bigint operator%(bigint a, ll b) {
    bigint div_res = a / b;
    return a - div_res * b;
}

bigint qp(bigint a, ll n) {
    bigint res(1);
    while (n) {
        if (n & 1) res = res * a;
        a = a * a;
        n >>= 1;
    }
    return res;
}

bigint comb(bigint n, bigint m) {
    bigint res = 1;
    for (bigint up = n, down = 1; down <= m; up = up - 1, down = down + 1)
        res = res * up, res = res / down;
    return res;
}

//快速comb

//n!中p的个数
int get(int n, int p) {
    int s = 0;
    while (n) {
        s += n / p;
        n /= p;
    }
    return s;
}

//c(n,m)中p的数量
int getC(int n, int m, int p) {
    return get(n, p) - get(m, p) - get(n - m, p);
}

vi prime;
bigint qcomb(int n, int m) {
    bigint res = 1;
    for (auto i : prime) {
        int x = getC(n, m, i);
        res = res * qp(i, x);
    }
    return res;
}



```

### int128

```c++
istream& operator>>(istream& cin, __int128& a) {
    a = 0;
    string scan;
    cin >> scan;
    for (int i = 0; i < scan.size(); i++) {
        a *= 10;
        a += scan[i] - '0';
    }
    return cin;
}
ostream& operator<<(ostream& cout, __int128 a) {
    if (a < 0) {
        cout << "-";
        a = -a;
    }
    if (a > 9)
        cout << a / 10;
    cout << (char)(a % 10 + '0');
    return cout;
}
```



### Gosper's hack

```c++
//枚举全集U=(1<<n)-1的所有1的个数恰好为k的子集 
void GosperHack(int k, int n) {
    int now = (1 << k) - 1;
    int lim = (1 << n);
    while (now < lim) {
        //do something  每个now即为所需的子集
        int lb = lowbit(now);
        int r = now + lb;
        now = (((r ^ now) >> 2) / lb) | r;
    }
}
```



### 拉格朗日插值

```
int n, k;//生成 1^k+2^k+...+n^k
ll F[N], mul;
int Finv[N], inv[N];//阶乘逆元,逆元
ll pre[N], suf[N];
int pcnt[N];//质因子个数
int f[N];
vi isp(N, 1), p;

//拉格朗日插值(一般用于求自然数幂次和)
void init() {
    //递推求逆元
    inv[1] = 1;
    for (int i = 2;i <= k + 2;i++) {
        inv[i] = (MOD - MOD / i) * 1ll * inv[MOD % i] % MOD;
    }
    //阶乘逆元
    Finv[0] = 1;
    for (int i = 1;i <= k + 2;i++) {
        Finv[i] = Finv[i - 1] * 1ll * inv[i] % MOD;
    }
    f[1] = 1;
    for (int i = 2;i <= k + 2;i++) {
        if (isp[i]) {
            p.push_back(i), f[i] = qp(i, k), pcnt[i] = 1;
        }
        for (auto j : p) {
            if (j > (k + 2) / i) {
                break;
            }
            isp[i * j] = 0;
            if (i % j == 0) {
                pcnt[i * j] = pcnt[i] + 1;
                f[i * j] = 1ll * f[i] * f[j] % MOD;
                break;
            }
            else {
                pcnt[i * j] = 1;
                f[i * j] = 1ll * f[i] * f[j] % MOD;
            }
        }
    }
    for (int i = 1;i <= k + 2;i++) {
        F[i] = (F[i - 1] + f[i]) % MOD;
    }
}
int norm(int x) {
    return (x % MOD + MOD) % MOD;
}
void Solve(int TIME) {

    cin >> n >> k;
    init();
    if (n <= k + 2) {
        cout << F[n] << endl;
        return;
    }
    pre[0] = suf[k + 3] = 1;
    for (int i = 1;i <= k + 2;i++) {
        pre[i] = pre[i - 1] * (n - i) % MOD;
    }
    for (int i = k + 2;i >= 1;i--) {
        suf[i] = suf[i + 1] * (n - i) % MOD;
    }
    int res = 0;
    for (int i = 1;i <= k + 2;i++) {
        int t = ((k + 2 - i) & 1) ? -1 : 1;
        int M = Finv[i - 1] * Finv[k + 2 - i] % MOD;
        int Z = pre[i - 1] * suf[i + 1] % MOD;
        res += F[i] * M % MOD * Z % MOD * t;
        res = norm(res);
    }
    cout << res << endl;
}
```



### 01分数规划

给定两个数组，$a_i$表示选取$i$的收益，$b_i$表示选取$i$的代价，定义$x[i]=1$或$0$，表示选或不选，每种物品只有选或不选两种方案，求出$\frac{\sum\limits_{i=1}^{n}a_i x_i}{\sum\limits_{i=1}^{n}b_i x_i }$的最大值/最小值

#### 二分

假设我们求最大值，二分一个答案$mid$，当存在一组$x_i$，满足$\frac{\sum\limits_{i=1}^{n}a_i x_i}{\sum\limits_{i=1}^{n}b_i x_i }>mid$，则说明当前的$mid$可以。

$\frac{\sum\limits_{i=1}^{n}a_i x_i}{\sum\limits_{i=1}^{n}b_i x_i }>mid \;⇒\; \sum\limits_{i=1}^{n}a_ix_i-mid\sum\limits_{i=1}^{n}b_ix_i>0 \;⇒\;\sum\limits_{i=1}^{n}x_i(a_i-mid \cdot b_i)>0$

只要求出上述式子的最大值就可以了，如果最大值比$0$大，说明$mid$是可行的，否则不可行。

如果要求最小值，只要比$0$​小即可。

```c++
//二分求最小值
const ld eps = 1e-8;
ld l = eps, r = 1e9;
while (r - l > eps) {
    ld mid = (l + r) / 2;
    vc<ld> g;
    for (auto [x, y, a, b] : edge) {
        g.push_back(a - b * mid);
    }
    sort(g.begin(), g.end());
    ld res = 0;
    for (int i = 0;i < k;i++) {//选k个物品
        res += g[i];
    }
    if (res < eps) r = mid;//ans
    else l = mid;
}


//二分求最大值
const ld eps = 1e-8;
ld l = eps, r = 1e9;
while (r - l > eps) {
    ld mid = (l + r) / 2;
    vc<ld> g;
    for (auto [x, y, a, b] : edge) {
        g.push_back(-(a - b * mid));
    }
    sort(g.begin(), g.end());
    ld res = 0;
    for (int i = 0;i < k;i++) {//取k个物品
        res += g[i];
    }
    if (res > eps) l = mid;//ans
    else r = mid;
}
```



#### $Dinkelbach$算法

思想是每次用上一轮的答案当做新的$mid$来输入，不断地迭代，直至答案满足精度。比二分更快。

```c++
//求最小值
const ld eps = 1e-8;
ld ans = 1e9;
while (1) {
    DSU d(n + 1);
    vc<pair<ld, ai2>> g;
    for (auto [x, y, a, b] : edge) {
        g.push_back({ a - b * ans,{a,b} });
    }
    sort(g.begin(), g.end());
    for (int i = 0;i < k;i++)//取k个物品
        A += g[i].second.at(0), B += g[i].second.at[1];
    ld t = 1.0 * A / B;
    if (abs(t - ans) < eps) break;
    ans = t;
}


//求最大值
const ld eps = 1e-8;
ld ans = eps;
while (1) {
    DSU d(n + 1);
    vc<pair<ld, ai2>> g;
    for (auto [x, y, a, b] : edge) {
        g.push_back({ -(a - b * ans),{a,b} });
    }
    sort(g.begin(), g.end());
    for (int i = 0;i < k;i++)//取k个物品
        A += g[i].second.at(0), B += g[i].second.at[1];
    ld t = 1.0 * A / B;
    if (abs(t - ans) < eps) break;
    ans = t;
}
```





### PBDS

```c++
#include<bits/extc++.h>
using namespace __gnu_pbds;
template <class T>
struct RBtree {
    using Tree = tree<T, null_type, less<T>, rb_tree_tag,
        tree_order_statistics_node_update>;
    Tree t;
    RBtree() {}
    RBtree(initializer_list<T> v) { for (auto x : v) { t.insert(x); } }
    int size() { return t.size(); }
    bool empty() { return t.empty(); }
    void insert(T x) { t.insert(x); }
    void erase(T x) { t.erase(t.lower_bound(x)); }
    T lower_bound(T x) { return t.lower_bound(x); }
    T upper_bound(T x) { return t.upper_bound(x); }
    T prev(T x) { auto it = t.lower_bound(x);it--;return *it; }
    T next(T x) { return *t.upper_bound(x); }
    int rank(T x) { return t.order_of_key(x) + 1; } //x的排名,严格小于x的个数+1
    //排名为1<=k<=size的值
    T operator[](int k) const {
        auto it = t.find_by_order(k - 1);
        if (it != t.end()) return *it;
        return T();//非法
    }
};
```





### 模拟退火

```c++
//mt19937 rng(random_device{}());
namespace SimulateAnneal {//模拟退火
    const ld MAX_TIME = 0.8;
    const ld EPS = 1e-14;
    ld RES;//最终答案
    ld RAND() { return rand() * 2 - RAND_MAX; }//随机函数
    //ld __RAND() { return 2ll * rng() - mt19937::max(); }//随机函数
    ld calc(ld x, ld y, ld z) {//所求函数

    }
    ld x, y, z;//函数的参数
    void SimulateAnneal() {
        ld T = 1e5;//初始温度
        const ld T0 = 0.9982;//降温速度
        while (T > EPS) {
            ld nx = x + RAND() * T;
            ld ny = y + RAND() * T;
            ld nz = z + RAND() * T;
            ld nRES = calc();
            ld delta = nRES - RES;
            if (delta < 0) {//更优则接受
                x = nx, y = ny, z = nz;
                RES = nRES;
            }
            else if (exp(-delta / T) * RAND_MAX > rand()) {//否则概率接受
                x = nx, y = ny, z = nz;
            }
            T = T * T0;
        }
    }

    void run() {
        RES = 1e100;
        for (int i = 1;i <= 5;i++) SimulateAnneal();
        //while ((double)clock() / CLOCKS_PER_SEC < MAX_TIME) SimulateAnneal();
    }
}
using namespace SimulateAnneal;
//srand(99999989);
```



### 将某个区间内所有数^x并得到结果集合

```cpp
struct seg
{
	vector<pair<int,int>>segs;
	const int MAN=(1<<30)-1;
	void get(int l,int r,int x){
		int len=r-l+1;
		int ll=l^(x&(~(len-1)));
		int rr=ll+len-1;
		segs.push_back({ll,rr});
	}
	void add(int l,int r,int ql,int qr,int x){
		if(l>=ql&&r<=qr){
			get(l,r,x);
		}
		int mid=l+r>>1;
		if(ql<=mid)add(l,mid,ql,qr,x);
		if(qr>mid)add(mid+1,r,ql,qr,x);
	}
	void add(int l,int r,int x){
		add(0,MAN,l,r,x);
	}
};
```



### (1)LCA

```c++
struct RMQ_LCA {
    int n, idx;
    vector<int> dfn;
    vector<vector<int>> adj;
    inline static int f[21][N];
    int getmin(int x, int y) { return dfn[x] < dfn[y] ? x : y; }

    RMQ_LCA(int n, vector<vector<int>>& adj, int root) :n(n), adj(adj), idx(0) {
        dfn.resize(n + 1);
        dfs(root);
        for (int j = 1, lgn = __lg(n); j <= lgn; j++) {
            for (int i = 1; i + (1 << j) - 1 <= n; i++) {
                f[j][i] = getmin(f[j - 1][i], f[j - 1][i + (1 << (j - 1))]);
            }
        }
    }
    ~RMQ_LCA() {
        for (int j = 0, lgn = __lg(n);j <= lgn;j++) {
            for (int i = 0;i <= n;i++) f[j][i] = 0;
        }
    }

    void dfs(int u, int p = 0) {
        f[0][dfn[u] = ++idx] = p;
        for (auto v : adj[u]) if (v != p) dfs(v, u);
    }

    int query(int l, int r) {
        int len = __lg(r - l + 1);
        return getmin(f[len][l], f[len][r - (1 << len) + 1]);
    }

    int lca(int u, int v) {
        if (u == v) return u;
        u = dfn[u], v = dfn[v];
        if (u > v) swap(u, v);
        return query(u + 1, v);
    }

};

```



### 幺半群滑窗器

```c++
template<typename T> struct SlidingWindowAggregation {
    stack<T> stack_rev, stack_pos;//序列逆序，序列正序
    stack<T> aux_stack_rev, aux_stack_pos;//辅助栈,维护序列逆序以及正序的运算前缀和
    T e;//单位元
    T e_rev, e_pos;//当前运算前缀和
    SlidingWindowAggregation(T e, function<T(T, T)> op) : e(e), e_rev(e), e_pos(e), op(op) {}

    int sz = 0;
    function<T(T, T)> op;//定义一种幺半群运算(满足可结合律,且有单位元)
    void push(T val) {
        if (stack_rev.empty()) {
            push_rev(val);
            transfer();
        }
        else {
            push_pos(val);
        }
        sz++;
    }

    void popleft() {
        if (sz == 0) return;
        if (stack_rev.empty()) transfer();
        stack_rev.pop();
        aux_stack_rev.pop();
        e_rev = aux_stack_rev.empty() ? e : aux_stack_rev.top();
        sz--;
    }

    T query() {
        return op(e_rev, e_pos);
    }

    int size() { return sz; }

    void push_rev(T val) {
        stack_rev.push(val);
        e_rev = op(val, e_rev);
        aux_stack_rev.push(e_rev);
    }

    void push_pos(T val) {
        stack_pos.push(val);
        e_pos = op(e_pos, val);
        aux_stack_pos.push(e_pos);
    }

    void transfer() {
        while (stack_pos.size()) {
            push_rev(stack_pos.top());
            stack_pos.pop();
        }
        while (aux_stack_pos.size()) {
            aux_stack_pos.pop();
        }
        e_pos = e;
    }
};
```



### 中位数对顶堆

```c++
 struct medianheap {
    multiset<int> small, big;
    ll sumsmall = 0, sumbig = 0;
    medianheap() {}
    void norm() {
        while (big.size() > small.size()) {
            int x = *big.begin();
            small.insert(x);
            big.erase(big.begin());
            sumsmall += x;
            sumbig -= x;
        }
        while (small.size() > big.size()) {
            int x = *small.rbegin();
            big.insert(x);
            small.erase(prev(small.end()));
            sumbig += x;
            sumsmall -= x;
        }
    }
    void insert(int x) {
        if (big.empty() || x >= *big.begin()) {
            big.insert(x);
            sumbig += x;
        }
        else {
            small.insert(x);
            sumsmall += x;
        }
        norm();
    }
    void erase(int x) {
        if (auto it = small.find(x); it != small.end()) {
            small.erase(it);
            sumsmall -= x;
        }
        else {
            big.erase(big.find(x));
            sumbig -= x;
        }
        norm();
    }
    int get() {
        assert(!big.empty());
        int median = *big.begin();
        int sz = small.size() + big.size();
        return sumbig - sumsmall - ((sz & 1) ? median : 0);
    }
};
```



