import boto3
from boto3.exceptions import ResourceNotExistsError
from botocore.exceptions import ValidationError
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime
from decimal import Decimal
import json
import pytz

dynamodb = boto3.resource("dynamodb")


def put_batch(table_name, data):
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")

    try:
        with table.batch_writer() as batch:
            for item in data:
                if "uuid" not in item.keys():
                    raise ValueError("Item must have a uuid")
                item_put = json.loads(json.dumps(item), parse_float=Decimal)
                batch.put_item(Item=item_put)
                print(f"Inserted item {item}")
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")


def put_item(table_name, item):
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")
    try:
        if "uuid" not in item.keys():
            raise ValueError("Item must have a uuid")
        item_put = json.loads(json.dumps(item), parse_float=Decimal)
        table.put_item(Item=item_put)
        print(f"Inserted item {item}")
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")


def clean_table(table_name):
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")
    try:
        with table.batch_writer() as batch:
            for item in table.scan()["Items"]:
                batch.delete_item(Key={"uuid": item["uuid"]})
                print(f"Deleted item {item}")
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")


def get_items(table_name):
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")

    try:
        items = table.scan()["Items"]
        return items
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")


def get_item(table_name, uuid):
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")

    try:
        item = table.get_item(Key={"uuid": uuid})["Item"]
        return item
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")


def get_item_last_24h(table_name, uuid):
    # pubMillis es la hora de publicación de la alerta, está en milisegundos y UTC
    try:
        table = dynamodb.Table(table_name)
    except ResourceNotExistsError:
        raise ValueError(f"Table {table_name} does not exist")

    try:
        # Set UTC
        datetime.now().astimezone(pytz.utc)
        item = table.query(
            KeyConditionExpression=Key("uuid").eq(uuid),
            FilterExpression=Attr("pubMillis").gt(
                datetime.now().timestamp() - 24 * 60 * 60 * 1000
            ),
            ScanIndexForward=False,
        )["Items"][0]
        return item
    except ValidationError as e:
        raise ValueError(f"Validation Error: {e}")
