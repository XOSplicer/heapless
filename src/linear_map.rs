//! A fixed capacity map/dictionary that performs lookups via linear search.
//!
//! Note that as this map doesn't use hashing so most operations are *O*(n) instead of *O*(1).

use core::{borrow::Borrow, fmt, mem, ops, slice};

use crate::{
    storage::{OwnedStorage, Storage, ViewStorage},
    vec::VecInner,
    Vec,
};

/// Base struct for [`LinearMap`] and [`LinearMapView`]
pub struct LinearMapInner<K, V, S: Storage> {
    pub(crate) buffer: VecInner<(K, V), S>,
}

/// A fixed capacity map/dictionary that performs lookups via linear search.
///
/// Note that as this map doesn't use hashing so most operations are *O*(n) instead of *O*(1).
pub type LinearMap<K, V, const N: usize> = LinearMapInner<K, V, OwnedStorage<N>>;

/// A dynamic capacity map/dictionary that performs lookups via linear search.
///
/// Note that as this map doesn't use hashing so most operations are *O*(n) instead of *O*(1).
pub type LinearMapView<K, V> = LinearMapInner<K, V, ViewStorage>;

impl<K, V, const N: usize> LinearMap<K, V, N> {
    /// Creates an empty `LinearMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut map: LinearMap<&str, isize, 8> = LinearMap::new();
    ///
    /// // allocate the map in a static variable
    /// static mut MAP: LinearMap<&str, isize, 8> = LinearMap::new();
    /// ```
    pub const fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Get a reference to the `LinearMap`, erasing the `N` const-generic.
    pub fn as_view(&self) -> &LinearMapView<K, V> {
        self
    }

    /// Get a mutable reference to the `LinearMap`, erasing the `N` const-generic.
    pub fn as_mut_view(&mut self) -> &mut LinearMapView<K, V> {
        self
    }
}

impl<K, V, S: Storage> LinearMapInner<K, V, S>
where
    K: Eq,
{
    /// Returns the number of elements that the map can hold.
    ///
    /// Computes in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let map: LinearMap<&str, isize, 8> = LinearMap::new();
    /// assert_eq!(map.capacity(), 8);
    /// ```
    pub fn capacity(&self) -> usize {
        self.buffer.storage_capacity()
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// Computes in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert(1, "a").unwrap();
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.buffer.clear()
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// Computes in *O*(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert(1, "a").unwrap();
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// Computes in *O*(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert(1, "a").unwrap();
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.iter()
            .find(|&(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// Computes in *O*(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert(1, "a").unwrap();
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.iter_mut()
            .find(|&(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    /// Returns the number of elements in this map.
    ///
    /// Computes in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut a: LinearMap<_, _, 8> = LinearMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a").unwrap();
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old value is returned.
    ///
    /// Computes in *O*(n) time
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// assert_eq!(map.insert(37, "a").unwrap(), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b").unwrap();
    /// assert_eq!(map.insert(37, "c").unwrap(), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert(&mut self, key: K, mut value: V) -> Result<Option<V>, (K, V)> {
        if let Some((_, v)) = self.iter_mut().find(|&(k, _)| *k == key) {
            mem::swap(v, &mut value);
            return Ok(Some(value));
        }

        self.buffer.push((key, value))?;
        Ok(None)
    }

    /// Returns true if the map contains no elements.
    ///
    /// Computes in *O*(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut a: LinearMap<_, _, 8> = LinearMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a").unwrap();
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert("a", 1).unwrap();
    /// map.insert("b", 2).unwrap();
    /// map.insert("c", 3).unwrap();
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            iter: self.buffer.as_slice().iter(),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert("a", 1).unwrap();
    /// map.insert("b", 2).unwrap();
    /// map.insert("c", 3).unwrap();
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val = 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            iter: self.buffer.as_mut_slice().iter_mut(),
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert("a", 1).unwrap();
    /// map.insert("b", 2).unwrap();
    /// map.insert("c", 3).unwrap();
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Removes a key from the map, returning the value at
    /// the key if the key was previously in the map.
    ///
    /// Computes in *O*(n) time
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert(1, "a").unwrap();
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let idx = self
            .keys()
            .enumerate()
            .find(|&(_, k)| k.borrow() == key)
            .map(|(idx, _)| idx);

        idx.map(|idx| self.buffer.swap_remove(idx).1)
    }

    /// An iterator visiting all values in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert("a", 1).unwrap();
    /// map.insert("b", 2).unwrap();
    /// map.insert("c", 3).unwrap();
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// An iterator visiting all values mutably in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// let mut map: LinearMap<_, _, 8> = LinearMap::new();
    /// map.insert("a", 1).unwrap();
    /// map.insert("b", 2).unwrap();
    /// map.insert("c", 3).unwrap();
    ///
    /// for val in map.values_mut() {
    ///     *val += 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }
}

impl<'a, K, V, Q, S: Storage> ops::Index<&'a Q> for LinearMapInner<K, V, S>
where
    K: Borrow<Q> + Eq,
    Q: Eq + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<'a, K, V, Q, S: Storage> ops::IndexMut<&'a Q> for LinearMapInner<K, V, S>
where
    K: Borrow<Q> + Eq,
    Q: Eq + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut V {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K, V, const N: usize> Default for LinearMap<K, V, N>
where
    K: Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const N: usize> Clone for LinearMap<K, V, N>
where
    K: Eq + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

impl<K, V, S: Storage> fmt::Debug for LinearMapInner<K, V, S>
where
    K: Eq + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, const N: usize> FromIterator<(K, V)> for LinearMap<K, V, N>
where
    K: Eq,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut out = Self::new();
        out.buffer.extend(iter);
        out
    }
}

/// An iterator that moves out of a [`LinearMap`].
///
/// This struct is created by calling the [`into_iter`](LinearMap::into_iter) method on [`LinearMap`].
pub struct IntoIter<K, V, const N: usize>
where
    K: Eq,
{
    inner: <Vec<(K, V), N> as IntoIterator>::IntoIter,
}

impl<K, V, const N: usize> Iterator for IntoIter<K, V, N>
where
    K: Eq,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<K, V, const N: usize> IntoIterator for LinearMap<K, V, N>
where
    K: Eq,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.buffer.into_iter(),
        }
    }
}

impl<'a, K, V, S: Storage> IntoIterator for &'a LinearMapInner<K, V, S>
where
    K: Eq,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over the items of a [`LinearMap`]
///
/// This struct is created by calling the [`iter`](LinearMap::iter) method on [`LinearMap`].
#[derive(Clone, Debug)]
pub struct Iter<'a, K, V> {
    iter: slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        // False positive from clippy
        // Option<&(K, V)> -> Option<(&K, &V)>
        #[allow(clippy::map_identity)]
        self.iter.next().map(|(k, v)| (k, v))
    }
}

/// An iterator over the items of a [`LinearMap`] that allows modifying the items
///
/// This struct is created by calling the [`iter_mut`](LinearMap::iter_mut) method on [`LinearMap`].
#[derive(Debug)]
pub struct IterMut<'a, K, V> {
    iter: slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, v)| (k as &K, v))
    }
}

impl<K, V, S1: Storage, S2: Storage> PartialEq<LinearMapInner<K, V, S2>>
    for LinearMapInner<K, V, S1>
where
    K: Eq,
    V: PartialEq,
{
    fn eq(&self, other: &LinearMapInner<K, V, S2>) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S: Storage> Eq for LinearMapInner<K, V, S>
where
    K: Eq,
    V: PartialEq,
{
}

impl<K, V, S> LinearMapInner<K, V, S>
where
    S: Storage,
    K: Eq,
{
    /// Returns an entry for the corresponding key
    /// ```
    /// use heapless::LinearMap;
    /// use heapless::linear_map::Entry;
    /// let mut map = LinearMap::<_, _, 16>::new();
    /// if let Entry::Vacant(v) = map.entry("a") {
    ///     v.insert(1).unwrap();
    /// }
    /// if let Entry::Occupied(mut o) = map.entry("a") {
    ///     println!("found {}", *o.get()); // Prints 1
    ///     o.insert(2);
    /// }
    /// // Prints 2
    /// println!("val: {}", *map.get("a").unwrap());
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        let idx_opt = self
            .keys()
            .enumerate()
            .find(|&(_, k)| k == &key)
            .map(|(idx, _)| idx);
        match idx_opt {
            Some(idx) => Entry::Occupied(OccupiedEntry {
                key,
                idx,
                map: self,
            }),
            None => Entry::Vacant(VacantEntry { key, map: self }),
        }
    }
}

/// A view into an entry in the map
pub enum Entry<'a, K, V, S: Storage> {
    /// The entry corresponding to the key `K` exists in the map
    Occupied(OccupiedEntry<'a, K, V, S>),
    /// The entry corresponding to the key `K` does not exist in the map
    Vacant(VacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    S: Storage,
    K: Eq,
{
    /// Ensures a value is in the entry by inserting the default if empty, and
    /// returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<_, _, 16> = LinearMap::new();
    ///
    /// let result = book_reviews
    ///     .entry("Adventures of Huckleberry Finn")
    ///     .or_insert("My favorite book.");
    ///
    /// assert_eq!(result, Ok(&mut "My favorite book."));
    /// assert_eq!(
    ///     book_reviews["Adventures of Huckleberry Finn"],
    ///     "My favorite book."
    /// );
    /// ```
    pub fn or_insert(self, default: V) -> Result<&'a mut V, (K, V)> {
        match self {
            Self::Occupied(entry) => Ok(entry.into_mut()),
            Self::Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default
    /// function if empty, and returns a mutable reference to the value in the
    /// entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<_, _, 16> = LinearMap::new();
    ///
    /// let s = "Masterpiece.".to_string();
    ///
    /// book_reviews
    ///     .entry("Grimms' Fairy Tales")
    ///     .or_insert_with(|| s);
    ///
    /// assert_eq!(
    ///     book_reviews["Grimms' Fairy Tales"],
    ///     "Masterpiece.".to_string()
    /// );
    /// ```
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> Result<&'a mut V, (K, V)> {
        match self {
            Self::Occupied(entry) => Ok(entry.into_mut()),
            Self::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of
    /// the default function. This method allows for generating key-derived
    /// values for insertion by providing the default function a reference to
    /// the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying
    /// the key is unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<_, _, 16> = LinearMap::new();
    ///
    ///
    /// book_reviews
    ///     .entry("Pride and Prejudice")
    ///     .or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(book_reviews["Pride and Prejudice"], 19);
    /// ```
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> Result<&'a mut V, (K, V)> {
        match self {
            Self::Occupied(entry) => Ok(entry.into_mut()),
            Self::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<&str, &str, 16> = LinearMap::new();
    /// assert_eq!(
    ///     book_reviews
    ///         .entry("The Adventures of Sherlock Holmes")
    ///         .key(),
    ///     &"The Adventures of Sherlock Holmes"
    /// );
    /// ```
    pub fn key(&self) -> &K {
        match *self {
            Self::Occupied(ref entry) => entry.key(),
            Self::Vacant(ref entry) => entry.key(),
        }
    }

    /// Consumes this entry to yield to key associated with it
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<u32, &str, 16> = LinearMap::new();
    /// assert_eq!(
    ///     book_reviews
    ///         .entry(42)
    ///         .into_key(),
    ///     42
    /// );
    /// ```
    pub fn into_key(self) -> K {
        match self {
            Self::Occupied(entry) => entry.into_key(),
            Self::Vacant(entry) => entry.into_key(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<_, _, 16> = LinearMap::new();
    ///
    /// book_reviews
    ///     .entry("Grimms' Fairy Tales")
    ///     .and_modify(|e| *e = "Masterpiece.")
    ///     .or_insert("Very enjoyable.");
    /// assert_eq!(book_reviews["Grimms' Fairy Tales"], "Very enjoyable.");
    /// ```
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Self::Occupied(mut entry) => {
                f(entry.get_mut());
                Self::Occupied(entry)
            }
            Self::Vacant(entry) => Self::Vacant(entry),
        }
    }
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    S: Storage,
    K: Eq,
    V: Default,
{
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::LinearMap;
    ///
    /// // allocate the map on the stack
    /// let mut book_reviews: LinearMap<&str, Option<&str>, 16> = LinearMap::new();
    ///
    /// book_reviews.entry("Pride and Prejudice").or_default();
    ///
    /// assert_eq!(book_reviews["Pride and Prejudice"], None);
    /// ```
    #[inline]
    pub fn or_default(self) -> Result<&'a mut V, (K, V)> {
        match self {
            Self::Occupied(entry) => Ok(entry.into_mut()),
            Self::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

/// An occupied entry which can be manipulated
pub struct OccupiedEntry<'a, K, V, S: Storage> {
    key: K,
    // The index in the buffer where the key was found.
    // OccupiedEntry holds an exclusive reference to the map so this index is
    // valid for OccupiedEntry's lifetime and can be used to access the buffer
    // without checking bounds again.
    idx: usize,
    map: &'a mut LinearMapInner<K, V, S>,
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S>
where
    S: Storage,
    K: Eq,
{
    /// Gets a reference to the key that this entity corresponds to
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes this entry to yield to key associated with it
    pub fn into_key(self) -> K {
        self.key
    }

    /// Removes this entry from the map and yields its corresponding key and value
    pub fn remove_entry(self) -> (K, V) {
        // SAFETY: idx has been returned by find() and therefore is inbounds
        unsafe { self.map.buffer.swap_remove_unchecked(self.idx) }
    }

    /// Removes this entry from the map and yields its value
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Gets a reference to the value associated with this entry
    pub fn get(&self) -> &V {
        // SAFETY: idx has been returned by find() and therefore is inbounds
        unsafe { &self.map.buffer.get_unchecked(self.idx).1 }
    }

    /// Gets a mutable reference to the value associated with this entry
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: idx has been returned by find() and therefore is inbounds
        unsafe { &mut self.map.buffer.get_unchecked_mut(self.idx).1 }
    }

    /// Consumes this entry and yields a reference to the underlying value
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: idx has been returned by find() and therefore is inbounds
        unsafe { &mut self.map.buffer.get_unchecked_mut(self.idx).1 }
    }

    /// Overwrites the underlying map's value with this entry's value
    pub fn insert(self, value: V) -> V {
        mem::replace(self.into_mut(), value)
    }
}

/// A view into an empty slot in the underlying map
pub struct VacantEntry<'a, K, V, S: Storage> {
    key: K,
    map: &'a mut LinearMapInner<K, V, S>,
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S>
where
    S: Storage,
    K: Eq,
{
    /// Get the key associated with this entry
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes this entry to yield to key associated with it
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts this entry into to underlying map, yields a mutable reference to the inserted value.
    /// If the map is at capacity the value and key are returned instead.
    pub fn insert(self, value: V) -> Result<&'a mut V, (K, V)> {
        // NOTE: Already checked that no other entry exists
        self.map.buffer.push((self.key, value)).map(|()| unsafe {
            // SAFETY: We just pushed to this vec, so last_mut() will never return Err
            &mut self.map.buffer.last_mut().unwrap_unchecked().1
        })
    }
}

#[cfg(test)]
mod test {
    use static_assertions::assert_not_impl_any;

    use super::{Entry, LinearMap, Vec};

    // Ensure a `LinearMap` containing `!Send` keys stays `!Send` itself.
    assert_not_impl_any!(LinearMap<*const (), (), 4>: Send);
    // Ensure a `LinearMap` containing `!Send` values stays `!Send` itself.
    assert_not_impl_any!(LinearMap<(), *const (), 4>: Send);

    #[test]
    fn static_new() {
        static mut _L: LinearMap<i32, i32, 8> = LinearMap::new();
    }

    #[test]
    fn partial_eq() {
        {
            let mut a = LinearMap::<_, _, 1>::new();
            a.insert("k1", "v1").unwrap();

            let mut b = LinearMap::<_, _, 2>::new();
            b.insert("k1", "v1").unwrap();

            assert!(a == b);

            b.insert("k2", "v2").unwrap();

            assert!(a != b);
        }

        {
            let mut a = LinearMap::<_, _, 2>::new();
            a.insert("k1", "v1").unwrap();
            a.insert("k2", "v2").unwrap();

            let mut b = LinearMap::<_, _, 2>::new();
            b.insert("k2", "v2").unwrap();
            b.insert("k1", "v1").unwrap();

            assert!(a == b);
        }
    }

    #[test]
    fn drop() {
        droppable!();

        {
            let mut v: LinearMap<i32, Droppable, 2> = LinearMap::new();
            v.insert(0, Droppable::new()).ok().unwrap();
            v.insert(1, Droppable::new()).ok().unwrap();
            v.remove(&1).unwrap();
        }

        assert_eq!(Droppable::count(), 0);

        {
            let mut v: LinearMap<i32, Droppable, 2> = LinearMap::new();
            v.insert(0, Droppable::new()).ok().unwrap();
            v.insert(1, Droppable::new()).ok().unwrap();
        }

        assert_eq!(Droppable::count(), 0);
    }

    #[test]
    fn into_iter() {
        let mut src: LinearMap<_, _, 4> = LinearMap::new();
        src.insert("k1", "v1").unwrap();
        src.insert("k2", "v2").unwrap();
        src.insert("k3", "v3").unwrap();
        src.insert("k4", "v4").unwrap();
        let clone = src.clone();
        for (k, v) in clone.into_iter() {
            assert_eq!(v, src.remove(k).unwrap());
        }
    }

    #[test]
    fn entry_or_insert() {
        let mut a: LinearMap<_, _, 2> = LinearMap::new();
        a.entry("k1").or_insert("v1").unwrap();
        assert_eq!(a["k1"], "v1");

        a.entry("k2").or_insert("v2").unwrap();
        assert_eq!(a["k2"], "v2");

        let result = a.entry("k3").or_insert("v3");
        assert_eq!(result, Err(("k3", "v3")));
    }

    #[test]
    fn entry_or_insert_with() {
        let mut a: LinearMap<_, _, 2> = LinearMap::new();
        a.entry("k1").or_insert_with(|| "v1").unwrap();
        assert_eq!(a["k1"], "v1");

        a.entry("k2").or_insert_with(|| "v2").unwrap();
        assert_eq!(a["k2"], "v2");

        let result = a.entry("k3").or_insert_with(|| "v3");
        assert_eq!(result, Err(("k3", "v3")));
    }

    #[test]
    fn entry_or_insert_with_key() {
        let mut a: LinearMap<_, _, 2> = LinearMap::new();
        a.entry("k1")
            .or_insert_with_key(|key| key.chars().count())
            .unwrap();
        assert_eq!(a["k1"], 2);

        a.entry("k22")
            .or_insert_with_key(|key| key.chars().count())
            .unwrap();
        assert_eq!(a["k22"], 3);

        let result = a.entry("k3").or_insert_with_key(|key| key.chars().count());
        assert_eq!(result, Err(("k3", 2)));
    }

    #[test]
    fn entry_key() {
        let mut a: LinearMap<&str, &str, 2> = LinearMap::new();

        assert_eq!(a.entry("k1").key(), &"k1");
    }

    #[test]
    fn entry_and_modify() {
        let mut a: LinearMap<&str, &str, 2> = LinearMap::new();
        a.insert("k1", "v1").unwrap();
        a.entry("k1").and_modify(|e| *e = "modified v1");

        assert_eq!(a["k1"], "modified v1");

        a.entry("k2")
            .and_modify(|e| *e = "v2")
            .or_insert("default v2")
            .unwrap();

        assert_eq!(a["k2"], "default v2");
    }

    #[test]
    fn entry_or_default() {
        let mut a: LinearMap<&str, Option<u32>, 2> = LinearMap::new();
        a.entry("k1").or_default().unwrap();

        assert_eq!(a["k1"], None);

        let mut b: LinearMap<&str, u8, 2> = LinearMap::new();
        b.entry("k2").or_default().unwrap();

        assert_eq!(b["k2"], 0);
    }

    const MAP_SLOTS: usize = 64;
    fn almost_filled_map() -> LinearMap<usize, usize, MAP_SLOTS> {
        // Create the inner buffer directly to skip linear search on every insert
        let mut buffer: Vec<(usize, usize), MAP_SLOTS> = Vec::new();
        for i in 1..MAP_SLOTS {
            // SAFETY: buffer can hold MAP_SLOTS items, and MAP_SLOTS-1 are pushed
            unsafe {
                buffer.push_unchecked((i, i));
            }
        }
        LinearMap { buffer }
    }

    #[test]
    fn entry_find() {
        let key = 0;
        let value = 0;
        let mut src = almost_filled_map();
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(_) => {
                panic!("Found entry without inserting");
            }
            Entry::Vacant(v) => {
                assert_eq!(&key, v.key());
                assert_eq!(key, v.into_key());
            }
        }
        src.insert(key, value).unwrap();
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(mut o) => {
                assert_eq!(&key, o.key());
                assert_eq!(&value, o.get());
                assert_eq!(&value, o.get_mut());
                assert_eq!(&value, o.into_mut());
            }
            Entry::Vacant(_) => {
                panic!("Entry not found");
            }
        }
    }

    #[test]
    fn entry_vacant_insert() {
        let key = 0;
        let value = 0;
        let mut src = almost_filled_map();
        assert_eq!(MAP_SLOTS - 1, src.len());
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(_) => {
                panic!("Entry found when empty");
            }
            Entry::Vacant(v) => {
                assert_eq!(value, *v.insert(value).unwrap());
            }
        };
        assert_eq!(value, *src.get(&key).unwrap())
    }

    #[test]
    fn entry_occupied_insert() {
        let key = 0;
        let value = 0;
        let value2 = 5;
        let mut src = almost_filled_map();
        assert_eq!(MAP_SLOTS - 1, src.len());
        src.insert(key, value).unwrap();
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(o) => {
                assert_eq!(value, o.insert(value2));
            }
            Entry::Vacant(_) => {
                panic!("Entry not found");
            }
        };
        assert_eq!(value2, *src.get(&key).unwrap())
    }

    #[test]
    fn entry_remove_entry() {
        let key = 0;
        let value = 0;
        let mut src = almost_filled_map();
        src.insert(key, value).unwrap();
        assert_eq!(MAP_SLOTS, src.len());
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(o) => {
                assert_eq!((key, value), o.remove_entry());
            }
            Entry::Vacant(_) => {
                panic!("Entry not found")
            }
        };
        assert_eq!(MAP_SLOTS - 1, src.len());
    }

    #[test]
    fn entry_remove() {
        let key = 0;
        let value = 0;
        let mut src = almost_filled_map();
        src.insert(key, value).unwrap();
        assert_eq!(MAP_SLOTS, src.len());
        let entry = src.entry(key);
        match entry {
            Entry::Occupied(o) => {
                assert_eq!(value, o.remove());
            }
            Entry::Vacant(_) => {
                panic!("Entry not found");
            }
        };
        assert_eq!(MAP_SLOTS - 1, src.len());
    }

    #[test]
    fn entry_roll_through_all() {
        let mut src: LinearMap<usize, usize, MAP_SLOTS> = LinearMap::new();
        for i in 0..MAP_SLOTS {
            match src.entry(i) {
                Entry::Occupied(_) => {
                    panic!("Entry found before insert");
                }
                Entry::Vacant(v) => {
                    assert_eq!(i, *v.insert(i).unwrap());
                }
            }
        }
        let add_mod = 99;
        for i in 0..MAP_SLOTS {
            match src.entry(i) {
                Entry::Occupied(o) => {
                    assert_eq!(i, o.insert(i + add_mod));
                }
                Entry::Vacant(_) => {
                    panic!("Entry not found after insert");
                }
            }
        }
        for i in 0..MAP_SLOTS {
            match src.entry(i) {
                Entry::Occupied(o) => {
                    assert_eq!((i, i + add_mod), o.remove_entry());
                }
                Entry::Vacant(_) => {
                    panic!("Entry not found after insert");
                }
            }
        }
        for i in 0..MAP_SLOTS {
            assert!(matches!(src.entry(i), Entry::Vacant(_)));
        }
        assert!(src.is_empty());
    }
}
