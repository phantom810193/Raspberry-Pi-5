-- data.sql - seed 5 purchase rows
DROP TABLE IF EXISTS purchases;
CREATE TABLE purchases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user TEXT NOT NULL,
  item TEXT NOT NULL,
  amount REAL NOT NULL
);
INSERT INTO purchases(user, item, amount) VALUES ('alice', 'raspberry pi 5', 85.0);
INSERT INTO purchases(user, item, amount) VALUES ('bob', 'camera module', 25.0);
INSERT INTO purchases(user, item, amount) VALUES ('carol', 'heatsink', 8.5);
INSERT INTO purchases(user, item, amount) VALUES ('dave', 'sd card', 12.0);
INSERT INTO purchases(user, item, amount) VALUES ('erin', 'power supply', 9.9);
