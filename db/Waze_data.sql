CREATE TABLE "alerts" (
  "uuid" UUID PRIMARY KEY,
  "reliability" smallint,
  "type" varchar(20),
  "road_type" smallint,
  "magvar" real,
  "subtype" varchar(40),
  "location_id" int,
  "pub_millis" bigint,
  "end_pub_millis" bigint
);

CREATE TABLE "alerts_location" (
  "id" int PRIMARY KEY,
  "x" decimal(3,6),
  "y" decimal(3,6),
  "street" varchar(50),
  "segment" int
);

CREATE TABLE "jams" (
  "uuid" bigint PRIMARY KEY,
  "level" smallint,
  "speed_kmh" real,
  "length" smallint,
  "end_node" varchar(50),
  "road_type" smallint,
  "delay" smallint,
  "street" varchar(50),
  "pub_millis" bigint,
  "end_pub_millis" bigint
);

CREATE TABLE "jams_lines" (
  "id" int PRIMARY KEY,
  "jam_uuid" bigint,
  "position" smallint,
  "x" real,
  "y" real
);

CREATE TABLE "jams_segments" (
  "id" int PRIMARY KEY,
  "jam_uuid" bigint,
  "position" smallint,
  "segment_id" bigint,
  "from_node" bigint,
  "to_node" bigint,
  "is_forward" bool
);

ALTER TABLE "alerts" ADD FOREIGN KEY ("location_id") REFERENCES "alerts_location" ("id") ON DELETE CASCADE;

ALTER TABLE "jams_lines" ADD FOREIGN KEY ("jam_uuid") REFERENCES "jams" ("uuid") ON DELETE CASCADE;

ALTER TABLE "jams_segments" ADD FOREIGN KEY ("jam_uuid") REFERENCES "jams" ("uuid") ON DELETE CASCADE;
